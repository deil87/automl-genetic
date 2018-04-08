/*
package com.automl.classifier.ensemble.bagging

import scala.util.Random

import org.apache.spark.annotation.Experimental
import org.apache.spark.ml._
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.HasSeed
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{LongType, StructField, StructType}

/**
  * Params for [[Bagging]] and [[BaggingModel]].
  */
private[ml] trait BaggingParams[FeatureType, M <: PredictionModel[FeatureType, M]]
  extends PredictorParams with HasSeed {
  /**
    * Param for indicating whether bagged model is a classifier (true) or regressor (false).
    * This parameter affects how models are aggregated: voting is used for classification (with ties
    * broken arbitrarily) and averaging is used for regression.
    * Default: true (classification)
    * @group param
    */
  val isClassifier: BooleanParam = new BooleanParam(this, "isClassification",
    "indicates if bagged model is a classifier or regressor")

  /** @group getParam */
  def getIsClassifier: Boolean = $(isClassifier)

  /**
    * Param for number of bootstraped models.
    * Default: 3
    * @group param
    */
  val numModels: IntParam = new IntParam(this, "numModels",
    "number of models to train on bootstrapped samples (>=1)", ParamValidators.gtEq(1))

  /** @group getParam */
  def getNumModels: Int = $(numModels)

  setDefault(numModels-> 3, isClassifier->true)
}

/**
  * :: Experimental ::
  * Trains an ensemble of models using bootstrap aggregation. Given a dataset with N points,
  * the traditional bootstrap sample consists of N points sampled with replacement from the original
  * dataset. This class generates `numModels` bootstrap samples and uses `estimator` to train a model
  * on each sample. The predictions generated by the trained models are then aggregated to generate
  * the ensemble prediction.
  */
@Experimental
class Bagging[
FeatureType,
Learner <: Predictor[FeatureType, Learner, M],
M <: PredictionModel[FeatureType, M]](override val uid: String)
  extends Estimator[BaggingModel[FeatureType, M]]
    with BaggingParams[FeatureType, M]  {

  def this() = this(Identifiable.randomUID("bagging"))

  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)

  /** @group setParam */
  def setNumModels(value: Int): this.type = set(numModels, value)

  /** @group setParam */
  def setIsClassifier(value: Boolean): this.type = set(isClassifier, value)

  /** @group setParam */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  def setLabelCol(value: String): this.type = set(labelCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /**
    * Param for the [[predictor]] to be validated.
    * @group param
    */
  val predictor: Param[Predictor[FeatureType, Learner, M]] =
    new Param(this, "estimator", "estimator for bagging")

  /** @group getParam */
  def getPredictor: Predictor[FeatureType, Learner, M] = $(predictor)

  /** @group setParam */
  def setPredictor(value: Predictor[FeatureType, Learner, M]): this.type = set(predictor, value)

  override def fit(dataset: DataFrame): BaggingModel[FeatureType, M] = {
    Random.setSeed($(seed))
    val models = (0 until $(numModels)).map { _ =>
      val bootstrapSample = dataset.sample(true, 1.0, Random.nextLong())
      val x = $(predictor)
      $(predictor)
        .setFeaturesCol($(featuresCol))
        .setPredictionCol($(predictionCol))
        .setLabelCol($(labelCol))
        .fit(bootstrapSample)
    }
    copyValues(new BaggingModel[FeatureType, M](uid, models).setParent(this))
  }

  override def transformSchema(schema: StructType): StructType = {
    $(predictor).transformSchema(schema)
  }

  override def copy(extra: ParamMap): Bagging[FeatureType, Learner, M] = {
    val copied = defaultCopy(extra).asInstanceOf[Bagging[FeatureType, Learner, M]]
    if (copied.isDefined(predictor)) {
      copied.setPredictor(copied.getPredictor.copy(extra))
    }
    copied
  }
}

/**
  * :: Experimental ::
  * Model from bootstrap aggregating (bagging).
  *
  * TODO: type-safe way to ensure models has at least one
  */
@Experimental
class BaggingModel[FeatureType, M <: PredictionModel[FeatureType, M]] private[ml] (
                                                                                    override val uid: String,
                                                                                    val models: Seq[M])
  extends Model[BaggingModel[FeatureType, M]]
    with BaggingParams[FeatureType, M] {

  assert(models.size > 0,
    s"BaggingModel requires > 0 models to aggregate over, got ${models.size}")

  /** @group setParam */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  override def transform(dataset: DataFrame): DataFrame = {
    transformSchema(dataset.schema, logging = true)

    // these are constant across models since the estimator unchanged
    val predictionColName = $(predictionCol)
    val instanceIdColName = "instanceId"
    val modelIdColName = "modelId"
    val predictionCountsColName = "predCounts"

    val numberedDataset = Bagging.dfZipWithIndex(dataset, colName = instanceIdColName)
    val predictions = models.zipWithIndex.map { case (model, modelId: Int) =>
      val toModelId = udf { (x: Any) => modelId }
      // cast is required because of erasure
      model.asInstanceOf[M]
        .setFeaturesCol($(featuresCol))
        .setPredictionCol($(predictionCol))
        .transform(numberedDataset)
        .withColumn(modelIdColName, toModelId(col(instanceIdColName)))
    }.reduce[DataFrame] { case (a: DataFrame, b: DataFrame) =>
      a.unionAll(b)
    }
    val aggregatedPrediction = if (this.getIsClassifier) { // aggregate by voting
    // counts number of models voting for each (instance, prediction) pair
    val predictionCounts = predictions
        .groupBy(instanceIdColName, predictionColName)
        .agg(modelIdColName -> "count")
        .withColumnRenamed("count(" + modelIdColName + ")", predictionCountsColName)

      // gets the counts for the most predicted prediction
      val maxPredictionCounts = predictionCounts
        .groupBy(instanceIdColName)
        .agg(predictionCountsColName -> "max")
        .withColumnRenamed("max(" + predictionCountsColName + ")", predictionCountsColName)

      // join and project to recover actual prediction
      maxPredictionCounts
        .join(predictionCounts, Seq(instanceIdColName, predictionCountsColName))
        .drop(predictionCountsColName)
    } else { // aggregate by averaging
      predictions.groupBy(instanceIdColName)
        .agg(predictionColName -> "avg")
        .withColumnRenamed("avg(" + predictionColName + ")", predictionColName)
    }

    numberedDataset.join(aggregatedPrediction, instanceIdColName).drop(instanceIdColName)
  }

  override def transformSchema(schema: StructType): StructType = {
    models.head.transformSchema(schema)
  }

  override def copy(extra: ParamMap): BaggingModel[FeatureType, M] = {
    val copied = new BaggingModel[FeatureType, M](
      uid,
      models.map(_.copy(extra)))
    copyValues(copied, extra).setParent(parent)
  }
}

private object Bagging {
  def dfZipWithIndex(
                      df: DataFrame,
                      offset: Int = 1,
                      colName: String = "id",
                      inFront: Boolean = true) : DataFrame = {
    df.sqlContext.createDataFrame(
      df.rdd.zipWithIndex.map(ln =>
        Row.fromSeq(
          (if (inFront) Seq(ln._2 + offset) else Seq())
            ++ ln._1.toSeq ++
            (if (inFront) Seq() else Seq(ln._2 + offset))
        )
      ),
      StructType(
        (if (inFront) Array(StructField(colName, LongType, false)) else Array[StructField]()) ++
          df.schema.fields ++
          (if (inFront) Array[StructField]() else Array(StructField(colName, LongType, false)))
      )
    )
  }
}*/
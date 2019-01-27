package com.automl.template.simple

import com.automl.helper.FitnessResult
import com.automl.spark.SparkSessionProvider
import com.automl.template.{EvaluationMagnet, ModelKey}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.sql._

case class RandomForest() extends SimpleModelMember with SparkSessionProvider{
  override def name: String = "Random forest " + super.name

  override def modelKey: ModelKey = ModelKey("RandomForest")

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

  override def fitnessError(trainDF: DataFrame, testDF: DataFrame): FitnessResult = {

    //We can iterate over HLIST and use
    // new RandomForestRegressor().set()
    val rfr =  new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(trainDF.union(testDF))

    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, rfr))

    val model = pipeline.fit(trainDF)
    val predictions = model.transform(testDF)
      .drop("features")
      .withColumnRenamed("indexedFeatures", "features")

    val evaluator = new RegressionEvaluator()
    val rmse: Double = evaluator.evaluate(predictions)

    logger.info(s"$name : RMSE = " + rmse)
    FitnessResult(rmse, predictions)
  }

}

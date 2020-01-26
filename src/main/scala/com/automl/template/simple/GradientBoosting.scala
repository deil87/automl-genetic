package com.automl.template.simple

import com.automl.PaddedLogging
import com.automl.evolution.dimension.hparameter.{HyperParametersField, HyperParametersGroup, MutableHParameter}
import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.EvaluationMagnet
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.sql._

case class GradientBoosting()(implicit val logPaddingSize: Int = 0) extends SimpleModelMember with PaddedLogging{
  override def name: String = "Gradient boosting " + super.name

  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case BinaryClassificationProblem => true
    case MultiClassClassificationProblem => false
    case RegressionProblem => true
  }

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = null

  override def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType, hyperParametersField: Option[HyperParametersField]): FitnessResult = {

    problemType match {
      case RegressionProblem =>
        val gBTRegressor = new GBTRegressor()

        val pipeline = new Pipeline().setStages(Array(gBTRegressor))

        val pipelineModel = pipeline.fit(trainDF)

        val predictions = pipelineModel
          .transform(testDF)
          .cache()

        val evaluator = new RegressionEvaluator()

        val rmse = evaluator.evaluate(predictions)
        logger.info(s"$name : RMSE = " + rmse)
        FitnessResult(Map("rmse" -> rmse), problemType, predictions)

      case MultiClassClassificationProblem =>
        FitnessResult(???, ???, ???)
          // NOTE @note Multiclass labels are not currently supported by GBTClassifier

      case BinaryClassificationProblem =>

        // Index labels, adding metadata to the label column.
        // Fit on whole dataset to include all labels in index.
        val wholeData = trainDF.union(testDF)

//        wholeData.show(10)
        val labelIndexer = new StringIndexer()
          .setInputCol("label")
          .setOutputCol("indexedLabel")
          .setStringOrderType("alphabetAsc")
          .fit(wholeData)

        // Automatically identify categorical features, and index them.
        // Set maxCategories so features with > 4 distinct values are treated as continuous.
        val featureIndexer = new VectorIndexer()
          .setInputCol("features")
          .setOutputCol("indexedFeatures")
          .setMaxCategories(4)
          .fit(wholeData)

        val gbt = new GBTClassifier()
          .setLabelCol("indexedLabel")
          .setFeaturesCol("indexedFeatures")
          .setMaxIter(100)

        // Convert indexed labels back to original labels.
        val labelConverter = new IndexToString()
          .setInputCol("prediction")
          .setOutputCol("predictedLabel")
          .setLabels(labelIndexer.labels)

        // Chain indexers and GBT in a Pipeline.
        val pipeline = new Pipeline()
          .setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

        // Train model. This also runs the indexers.
        val model = pipeline.fit(trainDF)

        // Make predictions.
        val predictions = model.transform(testDF)

//        predictions.select("predictedLabel", "label", "features").show(5)

        // Select (prediction, true label) and compute test error.
        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMetricName("f1")

        val f1 = evaluator.evaluate(predictions)
        logger.info(s"$name : F1 = " + f1)
        FitnessResult(Map("f1" -> f1), problemType, predictions)
    }
  }

  //TODO move to constructor
  var hpGroupInternal: HyperParametersGroup[_ <: MutableHParameter[Double, _]] = null
}

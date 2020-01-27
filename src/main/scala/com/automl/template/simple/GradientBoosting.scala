package com.automl.template.simple

import com.automl.{LogLossCustom, PaddedLogging}
import com.automl.evolution.dimension.hparameter.{DecisionTreeHPGroup, HyperParametersField, HyperParametersGroup, MutableHParameter}
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

case class GradientBoosting(hpGroup: DecisionTreeHPGroup = DecisionTreeHPGroup())(implicit val logPaddingSize: Int = 0) extends SimpleModelMember with PaddedLogging{
  override def name: String = "Gradient boosting " + super.name

  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case BinaryClassificationProblem => true
    case MultiClassClassificationProblem => false
    case RegressionProblem => true
  }

  var hpGroupInternal: HyperParametersGroup[_ <: MutableHParameter[Double, _]] = hpGroup

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = null

  override def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType, hpFieldFromCoevolution: Option[HyperParametersField]): FitnessResult = {

    val activeHPGroup: HyperParametersGroup[_] = getRelevantHPGroupFromActiveHPField( hpFieldFromCoevolution).getOrElse(hpGroup)

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

      case MultiClassClassificationProblem => throw new UnsupportedOperationException()
          // NOTE @note Multiclass labels are not currently supported by GBTClassifier

      case BinaryClassificationProblem =>

        val gbt = new GBTClassifier()
          .setLabelCol("indexedLabel")
          .setFeaturesCol("features")
          .setMaxIter(5)


        // Train model. This also runs the indexers.
        val model = gbt.fit(trainDF)

        // Make predictions.
        val predictions = model.transform(testDF)

        predictions.show(5, false)

        // Select (prediction, true label) and compute test error.
        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("indexedLabel")
          .setPredictionCol("prediction")
          .setMetricName("f1")

        val f1 = evaluator.evaluate(predictions)

        val logLoss = LogLossCustom.compute(predictions)

        val mapOfMetrics = Map("f1" -> f1, "logloss" -> logLoss)
        info(s"Finished. ${name} ${activeHPGroup.hpParameters.mkString(",")} : ${mapOfMetrics.map{ nameToValue => nameToValue._1 + " = " + nameToValue._2}.mkString("",",", "")}. Number of rows = train:${trainDF.count()} / test:${testDF.count()}")

        FitnessResult(Map("f1" -> f1), problemType, predictions)
    }
  }

}

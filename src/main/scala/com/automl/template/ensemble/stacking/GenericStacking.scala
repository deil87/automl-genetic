package com.automl.template.ensemble.stacking

import com.automl.helper.FitnessResult
import com.automl.regressor.EnsemblingRegressor
import com.automl.template.{EvaluationMagnet, TemplateMember, TemplateTree, TreeContext}
import com.automl.template.ensemble.EnsemblingModelMember
import com.automl.classifier.ensemble.stacking.SparkGenericStacking
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.{Pipeline, PipelineStage, Predictor}
import org.apache.spark.sql.DataFrame
import utils.SparkMLUtils._


case class GenericStacking(unusedMetaLearner: PipelineStage = new LinearRegression()) extends StackingMember with LazyLogging {
  override def name: String = "SparkStacking " + super.name


  override def ensemblingFitnessError[A <: TemplateMember](trainDF: DataFrame,
                                                           testDF: DataFrame,
                                                           subMembers: Seq[TemplateTree[A]],
                                                           problemType: ProblemType)
                                                          (implicit tc: TreeContext = TreeContext()): FitnessResult = {
    logger.debug(s"Evaluating $name ...")
    val responseColumnName = if(problemType.isClassification) "indexedLabel" else "label"
    val stacking = new SparkGenericStacking(4, responseColumnName)

    stacking.foldingStage(trainDF, testDF)

    subMembers.foldLeft(stacking)((stackingModel, nextMember) => {

      stackingModel.addModel(nextMember, trainDF, testDF, problemType)
    })

    problemType match {
      case RegressionProblem =>
        val finalPredictions = stacking.performStacking(unusedMetaLearner)
          .select("uniqueIdColumn", "features", "prediction") //TODO make sure that performStacking is returning predictions for testDF

        val evaluator = new RegressionEvaluator()

        val predictionsReunitedWithLabels = finalPredictions.join(testDF.select("label", "uniqueIdColumn"), "uniqueIdColumn")

        val rmse = evaluator.evaluate(predictionsReunitedWithLabels)
        logger.info(s"Finished. $name RMSE Final:" + rmse)
        FitnessResult(Map("rmse" -> rmse), problemType, predictionsReunitedWithLabels)

      case MultiClassClassificationProblem | BinaryClassificationProblem =>
        import testDF.sparkSession.implicits._
        import org.apache.spark.sql.functions.rint

        val metaLearner = new LinearRegression().setFeaturesCol("features").setLabelCol("indexedLabel")

        val finalPredictions = stacking.performStacking(metaLearner)
          .showN_AndContinue(100, "Before rounding predictions from GenericStacking")
          .withColumnReplace("prediction", rint($"prediction")) //NOTE we need to round because LinearRegression metalearner returns raw predictions
          .select("uniqueIdColumn", "features", "prediction") //TODO make sure that performStacking is returning predictions for testDF
          .cache()

        val predictionsReunitedWithLabels = finalPredictions.join(testDF.select("indexedLabel", "uniqueIdColumn"), "uniqueIdColumn")
          .showN_AndContinue(100, "GenericStacking predictions")
          .cache()

        val evaluator = new MulticlassClassificationEvaluator()
          .setPredictionCol("prediction") //Prediction from Stacking
          .setLabelCol("indexedLabel") // True labels from testDF
          .setMetricName("f1")

        val f1 = evaluator.evaluate(predictionsReunitedWithLabels)
        logger.info(s"Finished. $name : F1 = " + f1)
        FitnessResult(Map("f1" -> f1), problemType, predictionsReunitedWithLabels)
    }
  }

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

}


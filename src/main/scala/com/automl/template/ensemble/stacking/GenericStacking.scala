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
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.DataFrame
import utils.SparkMLUtils._


case class GenericStacking(metaLearner: PipelineStage = new LinearRegression()) extends StackingMember with LazyLogging {
  override def name: String = "SparkStacking " + super.name


  override def ensemblingFitnessError[A <: TemplateMember](trainDF: DataFrame,
                                                           testDF: DataFrame,
                                                           subMembers: Seq[TemplateTree[A]],
                                                           problemType: ProblemType)
                                                          (implicit tc: TreeContext = TreeContext()): FitnessResult = {
    logger.debug(s"Evaluating $name ...")
    val stacking = new SparkGenericStacking(3)

    stacking.foldingStage(trainDF, testDF)

    subMembers.foldLeft(stacking)((stackingModel, nextMember) => {

      stackingModel.addModel(nextMember, trainDF, testDF, problemType: ProblemType)
    })

    val finalPredictions = stacking.performStacking(metaLearner)
      .select("uniqueIdColumn", "features", "prediction") //TODO make sure that performStacking is returning predictions for testDF
    //    logger.info("Final predictions (top 10) from GenericStacking:")
    //    finalPredictions.showN_AndContinue(10)

    problemType match {
      case RegressionProblem =>
        val evaluator = new RegressionEvaluator()

        val predictionsReunitedWithLabels = finalPredictions.join(testDF.select("label", "uniqueIdColumn"), "uniqueIdColumn")

        val rmse = evaluator.evaluate(predictionsReunitedWithLabels)
        logger.info("RMSE Final:" + rmse)
        FitnessResult(Map("rmse" -> rmse), problemType, predictionsReunitedWithLabels)
      case MultiClassClassificationProblem | BinaryClassificationProblem =>
        //TODO need to support classification case
        val predictionsReunitedWithLabels = finalPredictions.join(testDF.select("label", "uniqueIdColumn"), "uniqueIdColumn")


        val evaluator = new MulticlassClassificationEvaluator().setMetricName("f1")
        val f1 = evaluator.evaluate(predictionsReunitedWithLabels)
        logger.info(s"$name : F1 = " + f1)
        FitnessResult(Map("f1" -> f1), problemType, predictionsReunitedWithLabels)
    }
  }

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

}


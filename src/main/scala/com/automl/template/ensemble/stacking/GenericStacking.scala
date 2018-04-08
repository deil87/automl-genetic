package com.automl.template.ensemble.stacking

import com.automl.helper.FitnessResult
import com.automl.regressor.EnsemblingRegressor
import com.automl.template.{EvaluationMagnet, TemplateMember, TemplateTree, TreeContext}
import com.automl.template.ensemble.EnsemblingMember
import com.automl.classifier.ensemble.stacking.SparkGenericStacking
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.DataFrame
import utils.SparkMLUtils._


case class GenericStacking(metaLearner: PipelineStage = new LinearRegression()) extends StackingMember with LazyLogging {
  override def name: String = "SparkStacking " + super.name


  override def ensemblingFitnessError[A <: TemplateMember](trainDF: DataFrame,
                                                           testDF: DataFrame,
                                                           subMembers: Seq[TemplateTree[A]])
                                                          (implicit tc: TreeContext = TreeContext()): FitnessResult = {
    logger.info(s"\nEvaluating $name ...")
    val stacking = new SparkGenericStacking(3)

    stacking.foldingStage(trainDF, testDF)

    subMembers.foldLeft(stacking)((stackingModel, nextMember) => {

      stackingModel.addModel(nextMember, trainDF, testDF)
    })

    val finalPredictions = stacking.performStacking(metaLearner)
      .select("uniqueIdColumn", "features", "prediction") //TODO make sure that performStacking is returning predictions for testDF
    logger.info("Final predictions:")
    finalPredictions.showN_AndContinue(10)

    val evaluator = new RegressionEvaluator()

    val predictionsReunitedWithLabels = finalPredictions.join(testDF.select("label", "uniqueIdColumn"), "uniqueIdColumn")
    predictionsReunitedWithLabels.showN_AndContinue(10)

    val rmse = evaluator.evaluate(predictionsReunitedWithLabels)
    println("RMSE Final:" + rmse)
    FitnessResult(rmse, predictionsReunitedWithLabels)
  }

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

  override def fitnessError(trainDF: DataFrame, testDF: DataFrame): FitnessResult = ???

  override def ensemblingRegressor: EnsemblingRegressor = ??? //TODO Do we need it for all ensembling members?
}


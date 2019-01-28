package com.automl.template.simple

import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.EvaluationMagnet
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.sql._

case class GradientBoosting() extends SimpleModelMember with LazyLogging{
  override def name: String = "Gradient boosting " + super.name

  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case MultiClassClassificationProblem => true
    case RegressionProblem => true
  }

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

  override def fitnessError(trainDF: DataFrame, testDF: DataFrame): FitnessResult = {

    val gBTRegressor = new GBTRegressor()

    val pipeline = new Pipeline().setStages(Array(gBTRegressor))

    val pipelineModel = pipeline.fit(trainDF)

    val predictions = pipelineModel
      .transform(testDF)
      .cache()

    val evaluator = new RegressionEvaluator()

    val rmse = evaluator.evaluate(predictions)
    logger.info(s"$name : RMSE = " + rmse)
    FitnessResult(rmse, predictions)
  }

}

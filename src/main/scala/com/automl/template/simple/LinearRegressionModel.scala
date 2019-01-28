package com.automl.template.simple

import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.EvaluationMagnet
import com.automl.teststrategy.{TestStrategy, TrainingTestSplitStrategy}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql._

case class LinearRegressionModel() extends LinearModelMember with LazyLogging{

  override def name: String = "LinearRegressionModel " + super.name

  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case MultiClassClassificationProblem => false
    case RegressionProblem => true
  }

  override def testStrategy: TestStrategy = new TrainingTestSplitStrategy()


  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

  override def fitnessError(trainDF: DataFrame, testDF: DataFrame): FitnessResult = {
    logger.debug(s"Evaluating $name ...")
    val linearRegression = new LinearRegression() // It is a newer version of LinearRegressionWithSGD from mllib

    val model = linearRegression.fit(trainDF)
    val predictions = model.transform(testDF)

    predictions.cache()

    val evaluator = new RegressionEvaluator()

    val rmse: Double = evaluator.evaluate(predictions)

    logger.info(s"$name : RMSE = " + rmse)
    FitnessResult(rmse, predictions)
  }
}

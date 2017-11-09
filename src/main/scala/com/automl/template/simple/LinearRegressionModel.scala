package com.automl.template.simple

import com.automl.helper.FitnessResult
import com.automl.template.EvaluationMagnet
import com.automl.teststrategy.{TestStrategy, TrainingTestSplitStrategy}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql._
import org.apache.spark.sql.types.DoubleType
import utils.SparkMLUtils

case class LinearRegressionModel() extends SimpleModelMember {

  override def name: String = "LinearRegressionModel " + super.name

  override def testStrategy: TestStrategy = new TrainingTestSplitStrategy()


  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

  override def fitnessError(trainDF: DataFrame, testDF: DataFrame): FitnessResult = {

    val linearRegression = new LinearRegression()

    val model = linearRegression.fit(trainDF)
    val predictions = model.transform(testDF)

    predictions.cache()

    import SparkMLUtils._
    predictions.showN_AndContinue(10)

    val evaluator = new RegressionEvaluator()

    val rmse: Double = evaluator.evaluate(predictions)

    println(s"$name : RMSE = " + rmse)
    FitnessResult(rmse, predictions)
  }
}

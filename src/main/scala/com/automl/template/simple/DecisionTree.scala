package com.automl.template.simple

import com.automl.helper.FitnessResult
import com.automl.spark.SparkSessionProvider
import com.automl.template.EvaluationMagnet
import com.automl.teststrategy.{TestStrategy, TrainingTestSplitStrategy}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.sql._
import org.apache.spark.sql.types.DoubleType
import utils.SparkMLUtils

case class DecisionTree() extends SimpleModelMember with SparkSessionProvider with LazyLogging{
  override def name: String = "DecisionTree " + super.name


  override def testStrategy: TestStrategy = new TrainingTestSplitStrategy()


  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

  override def fitnessError(trainDF: DataFrame, testDF: DataFrame): FitnessResult = {
    logger.info(s"\nEvaluating $name ...")
    import  SparkMLUtils._

    import ss.implicits._


    val dtr = new DecisionTreeRegressor()

    val model = dtr.fit(trainDF)
    val predictions = model.transform(testDF)

    val evaluator = new RegressionEvaluator()

    val rmse: Double = evaluator.evaluate(predictions)

    logger.info(s"$name : RMSE = " + rmse)
    FitnessResult(rmse, predictions)
  }
}

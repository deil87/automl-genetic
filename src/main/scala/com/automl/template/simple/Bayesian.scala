package com.automl.template.simple

import com.automl.helper.FitnessResult
import com.automl.spark.SparkSessionProvider
import com.automl.template.EvaluationMagnet
import com.automl.teststrategy.{TestStrategy, TrainingTestSplitStrategy}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.sql._
import org.apache.spark.sql.types.DoubleType
import utils.SparkMLUtils

case class Bayesian() extends SimpleModelMember with SparkSessionProvider with LazyLogging{
  override def name: String = "Bayesian " + super.name

  override val testStrategy: TestStrategy = new TrainingTestSplitStrategy()

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

  override def fitnessError(trainingDF: DataFrame, testDF: DataFrame): FitnessResult = {
    logger.info(s"\nEvaluating $name ...")
    val nb = new NaiveBayes()


    val model = nb.fit(trainingDF)

    val predictions = model.transform(testDF)

    predictions.cache()

// TODO we can use   model.numClasses

    val evaluator = new RegressionEvaluator()

    import  SparkMLUtils._
    import ss.implicits._
    val rmse: Double = evaluator.evaluate(predictions.withColumnReplace("prediction", $"prediction".cast(DoubleType)))

    logger.info(s"$name : RMSE = " + rmse)
    FitnessResult(rmse, predictions.drop("rawPrediction").drop("probability"))
  }
}

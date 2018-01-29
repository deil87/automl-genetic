package com.automl.template.simple

import com.automl.helper.FitnessResult
import com.automl.template.EvaluationMagnet
import ml.dmlc.xgboost4j.scala.spark.XGBoostEstimator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql._

case class ExtreamGradientBoosting() extends SimpleModelMember {
  override def name: String = "ExtreamGradientBoosting " + super.name


  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

  override def fitnessError(trainDF: DataFrame, testDF: DataFrame): FitnessResult = {


    val paramsMap = Map[String, String](
      "objective" -> "reg:linear",
      "eval_metric" -> "rmse"
    )
    val xGBoostEstimator = new XGBoostEstimator(paramsMap)

    val pipeline = new Pipeline().setStages(Array(xGBoostEstimator))

    val pipelineModel = pipeline.fit(trainDF)


    val predictions = pipelineModel
      .transform(testDF)
      .cache()

    val evaluator = new RegressionEvaluator()

    val rmse = evaluator.evaluate(predictions)
    println(s"$name : RMSE = " + rmse)
    FitnessResult(rmse, predictions)
  }

}

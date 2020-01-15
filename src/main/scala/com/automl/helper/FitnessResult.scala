package com.automl.helper

import com.automl.ConfigProvider
import com.automl.evolution.evaluation.EvaluationContextInfo
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import org.apache.spark.sql.DataFrame

case class FitnessResult(metricsMap: Map[String, Double], problemType: ProblemType, dfWithPredictions: DataFrame) {

  private val config = ConfigProvider.config.getConfig("evolution")
  private lazy val multiclassMetric = config.getString("evaluation.multiclass.metric")

  def getCorrespondingMetric: Double = problemType match {
    case MultiClassClassificationProblem | BinaryClassificationProblem =>
      if(multiclassMetric == "logloss") metricsMap("logloss") else metricsMap("f1")
    case RegressionProblem =>
      metricsMap("rmse")
  }

  def dfWithPredictionsOnly: DataFrame = dfWithPredictions.select("uniqueIdColumn", "indexedLabel", "prediction")

  def getMetricByName(name: String) = metricsMap.getOrElse(name, throw new IllegalArgumentException(s"Metric with name $name was not found"))

  def compareTo(other: FitnessResult): Int =
    if (getCorrespondingMetric > other.getCorrespondingMetric) 1
    else if (getCorrespondingMetric < other.getCorrespondingMetric) -1
    else 0

  override def toString: String = problemType match {
    case MultiClassClassificationProblem | BinaryClassificationProblem =>
      if(multiclassMetric == "logloss") "logloss:" + metricsMap("logloss") else "f1:" + metricsMap("f1")

    case RegressionProblem =>
      "rmse:" + metricsMap("rmse")
  }
}

object FitnessResult {

  implicit val fitnessResultOrdering = new Ordering[FitnessResult] {
    override def compare(x: FitnessResult, y: FitnessResult) = {
      x.compareTo(y)
    }
  }
}


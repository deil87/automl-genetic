package com.automl.helper

import com.automl.{BetterThanComparable, ConfigProvider, EvaluatedTemplateData, EvaluationRules}
import com.automl.evolution.evaluation.EvaluationContextInfo
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import org.apache.spark.sql.DataFrame

case class FitnessResult(metricsMap: Map[String, Double], problemType: ProblemType, dfWithPredictions: DataFrame) extends BetterThanComparable[FitnessResult] with EvaluationRules{

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

  def betterThan(that:FitnessResult): Boolean = {
    betterThan(that, 1.0)
  }

  def betterThan(that:FitnessResult, coef: Double = 1.0): Boolean = {
    if(theBiggerTheBetter(problemType))
      compareTo(that, coef) > 0
    else compareTo(that, coef) < 0
  }

  def betterThanOrEqual(that:FitnessResult): Int = {
    if(theBiggerTheBetter(problemType))
      compareTo(that)
    else -compareTo(that)
  }

  def compareTo(that: FitnessResult, coef: Double = 1.0): Int =
    if (getCorrespondingMetric > that.getCorrespondingMetric * coef) 1
    else if (getCorrespondingMetric * coef < that.getCorrespondingMetric) -1
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
      x.betterThanOrEqual(y)
    }
  }
}


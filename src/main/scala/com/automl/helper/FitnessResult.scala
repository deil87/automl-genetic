package com.automl.helper

import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import org.apache.spark.sql.DataFrame

//TODO make it more generic as for now fitnessError is having only bad connotation but we are using it with metrics such as auc(and the higher the better)
//TODO there is an ambiguity that we use metrics and problem type. We can deduce problem type from set of metrics.
case class FitnessResult(metricsMap: Map[String, Double], problemType: ProblemType, dfWithPredictions: DataFrame) {
  def getCorrespondingMetric: Double = problemType match {
    case MultiClassClassificationProblem | BinaryClassificationProblem =>
      metricsMap("f1")
    case RegressionProblem =>
      metricsMap("rmse")
  }

  def dfWithPredictionsOnly: DataFrame = dfWithPredictions.select("uniqueIdColumn", "indexedLabel", "prediction")

  def getMetricByName(name: String) = metricsMap.getOrElse(name, throw new IllegalArgumentException(s"Metric with name $name was not found"))

  def orderTo(other: FitnessResult): Boolean = problemType match {
    case MultiClassClassificationProblem | BinaryClassificationProblem =>
      getCorrespondingMetric > other.getCorrespondingMetric
    case RegressionProblem =>
      getCorrespondingMetric < other.getCorrespondingMetric
  }

  def filterFun(other: FitnessResult): Boolean = ! orderTo(other)

  def compareTo(other: FitnessResult): Int = problemType match {
    case MultiClassClassificationProblem | BinaryClassificationProblem =>
      if(getCorrespondingMetric > other.getCorrespondingMetric) 1
      else if(getCorrespondingMetric < other.getCorrespondingMetric) -1
      else 0
    case RegressionProblem =>
      if(getCorrespondingMetric < other.getCorrespondingMetric) 1
      else if(getCorrespondingMetric > other.getCorrespondingMetric) -1
      else 0
  }
}

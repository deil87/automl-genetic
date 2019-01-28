package com.automl.problemtype
import org.apache.spark.sql.DataFrame

/**
  *  If number of distinct values is greater than threshold then it is a Regression Problem
  * @param threshold
  */
class ProblemTypeThresholdEstimator(threshold: Int) extends ProblemTypeEstimator {

  override def estimate(data: DataFrame, responseColumn: String): ProblemType = {
    val numberOfDistinctValues = data.select(responseColumn).distinct().take(threshold + 1).length
    if(numberOfDistinctValues > threshold) {
      ProblemType.RegressionProblem
    } else {
      ProblemType.MultiClassClassificationProblem
    }
  }
}

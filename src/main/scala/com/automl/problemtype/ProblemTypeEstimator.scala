package com.automl.problemtype

import org.apache.spark.sql.DataFrame

trait ProblemTypeEstimator {

  def estimate(data: DataFrame, responseColumn: String): ProblemType
}

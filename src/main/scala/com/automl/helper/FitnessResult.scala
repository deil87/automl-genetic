package com.automl.helper

import org.apache.spark.sql.DataFrame

case class FitnessResult(fitnessError: Double, dfWithPredictions: DataFrame)

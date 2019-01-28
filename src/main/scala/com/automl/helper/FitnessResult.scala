package com.automl.helper

import org.apache.spark.sql.DataFrame

//TODO make it more generic as for now fitnessError is having only bad connotation but we are using it with metrics such as auc(and the higher the better)
case class FitnessResult(fitnessError: Double, dfWithPredictions: DataFrame)

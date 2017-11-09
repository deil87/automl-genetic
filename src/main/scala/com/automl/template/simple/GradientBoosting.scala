package com.automl.template.simple

import com.automl.helper.FitnessResult
import com.automl.template.EvaluationMagnet
import org.apache.spark.sql._

case class GradientBoosting() extends SimpleModelMember {
  override def name: String = "Gradient boosting " + super.name

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

  override def fitnessError(trainDF: DataFrame, testDF: DataFrame): FitnessResult = ???

}

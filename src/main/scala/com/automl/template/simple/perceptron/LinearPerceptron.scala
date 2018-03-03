package com.automl.template.simple.perceptron

import com.automl.helper.FitnessResult
import com.automl.template.EvaluationMagnet
import com.automl.template.simple.SimpleModelMember
import org.apache.spark.sql._

case class LinearPerceptron() extends SimpleModelMember {
  override def name: String = "LinearPerceptron " + super.name

  lazy val predictor = ???

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

  override def fitnessError(trainDF: DataFrame, testDF: DataFrame): FitnessResult = {

    ???
  }

}

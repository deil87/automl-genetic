package com.automl.template.simple

import com.automl.helper.FitnessResult
import com.automl.template.EvaluationMagnet
import org.apache.spark.sql._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

case class NeuralNetwork() extends SimpleModelMember {
  override def name: String = "NeuralNetwork " + super.name

  lazy val predictor = new MultiLayerNetwork(???)

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

  override def fitnessError(trainDF: DataFrame, testDF: DataFrame): FitnessResult = ???

}

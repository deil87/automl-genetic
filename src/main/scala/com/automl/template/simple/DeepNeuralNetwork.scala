package com.automl.template.simple

import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.EvaluationMagnet
import org.apache.spark.sql._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

case class DeepNeuralNetwork() extends SimpleModelMember {
  override def name: String = "MultiLayerNetwork " + super.name

  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case MultiClassClassificationProblem => true
    case RegressionProblem => true
  }

  lazy val predictor = new MultiLayerNetwork(???)

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

  override def fitnessError(trainDF: DataFrame, testDF: DataFrame): FitnessResult = ???

}

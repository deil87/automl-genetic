package com.automl.template.simple

import com.automl.template._
import com.automl.teststrategy.TestStrategy
import org.apache.spark.sql._

trait SimpleModelMember extends TemplateMember {
  override def name: String = "simpleModel member"

  def modelKey: ModelKey = ???

  def testStrategy: TestStrategy = ???

}

object SimpleModelMember {
  val DeepNeuralNetwork: SimpleModelMember = new DeepNeuralNetwork()
  val NeuralNetwork: SimpleModelMember = new NeuralNetwork(Array(4,3,2,1))
  val Bayesian: SimpleModelMember = new Bayesian()
  val GLM: SimpleModelMember = new LinearRegressionModel()
  val DT: SimpleModelMember = new DecisionTree()
  val KNN: SimpleModelMember = new KNearestNeighbours()

  val poolOfSimpleModels: Seq[SimpleModelMember] = Seq(/*DeepNeuralNetwork,*/ Bayesian, GLM, DT/*, KNN*/)
}

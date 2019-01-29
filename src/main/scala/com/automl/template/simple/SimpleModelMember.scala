package com.automl.template.simple

import com.automl.problemtype.ProblemType
import com.automl.template._
import com.automl.template.simple.perceptron.LinearPerceptron
import com.automl.teststrategy.TestStrategy
import org.apache.spark.sql._

trait SimpleModelMember extends TemplateMember {
  override def name: String = "simpleModel member"

  def canHandleProblemType: PartialFunction[ProblemType, Boolean]

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
  val LogisticRegression: SimpleModelMember = new LogisticRegressionModel()
  val OneVsRestModel: SimpleModelMember = new OneVsRestModel()
  val SVM: SimpleModelMember = new SVMModel()
  val SupportVectorRegression: SimpleModelMember = new SupportVectorRegression()
  val LinearPerceptron: SimpleModelMember = new LinearPerceptron()
  val GradientBoosting: SimpleModelMember = new GradientBoosting() // Can we consider this as a simple model? It is actually an ensemble of trees

  val poolOfSimpleModels: Seq[SimpleModelMember] = Seq(/*DeepNeuralNetwork,*/ Bayesian, GLM, DT/*,GradientBoosting,  KNN, LogisticRegression, SVM, SupportVectorRegression, LinearPerceptron*/)

  def poolOfSimpleModels(problemType: ProblemType): Seq[SimpleModelMember] = {
    poolOfSimpleModels.filter(_.canHandleProblemType(problemType))
  }
}

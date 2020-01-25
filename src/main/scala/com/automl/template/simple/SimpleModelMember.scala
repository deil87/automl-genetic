package com.automl.template.simple

import com.automl.evolution.dimension.hparameter.{HyperParametersField, HyperParametersGroup, MutableHParameter}
import com.automl.{ConfigProvider, PaddedLogging}
import com.automl.problemtype.ProblemType
import com.automl.template._
import com.automl.template.simple.perceptron.LinearPerceptron
import com.automl.teststrategy.TestStrategy
import com.typesafe.config.Config

import scala.collection.JavaConverters._

trait SimpleModelMember extends TemplateMember { self: PaddedLogging =>
  override def name: String = "simpleModel member"

  def modelKey: ModelKey = ???

  def testStrategy: TestStrategy = ???
}

object SimpleModelMember {

  def DeepNeuralNetwork: SimpleModelMember = new DeepNeuralNetwork()
  def NeuralNetwork: SimpleModelMember = new NeuralNetwork(Array(4,3,2,1))
  def Bayesian: SimpleModelMember = new Bayesian()
  def GLM: SimpleModelMember = new LinearRegressionModel()
  def DT: SimpleModelMember = new DecisionTree()
  def RF: SimpleModelMember = new RandomForest()
  def KNN: SimpleModelMember = new KNearestNeighbours()
  def LogisticRegression: SimpleModelMember = new LogisticRegressionModel()
  def OneVsRestModel: SimpleModelMember = new OneVsRestModel()
  def SVM: SimpleModelMember = new SVMModel()
  def SupportVectorRegression: SimpleModelMember = new SupportVectorRegression()
  def LinearPerceptron: SimpleModelMember = new LinearPerceptron()
  def GradientBoosting: SimpleModelMember = new GradientBoosting() // Can we consider this as a simple model? It is actually an ensemble of trees

  def poolOfSimpleModels: Seq[SimpleModelMember] = Seq(/*DeepNeuralNetwork,*/ Bayesian, GLM, DT, RF/*, SVM*//*,GradientBoosting,  KNN, LogisticRegression, SupportVectorRegression, LinearPerceptron*/)

  def poolOfSimpleModels(problemType: ProblemType): Seq[SimpleModelMember] = {
    poolOfSimpleModels.filter(_.canHandleProblemType(problemType))
  }

  val tdConfig = ConfigProvider.config.getConfig("evolution.templateDimension")
  lazy val poolOfSimpleModelsNames: Seq[String] = tdConfig.getStringList("poolOfSimpleModels").asScala

  // Maybe it is better to specify names inside models so that we don't need to pattern match here and suplicate stringg ids
  def poolOfSimpleModelsByNames(names: Seq[String]): Seq[SimpleModelMember] = names.map {
    case "logistic_regression" => LogisticRegression
    case "decision_tree" => DT
    case "random_forest" => RF
    case "bayesian" => Bayesian
  }

  def randomMember(problemType: ProblemType): SimpleModelMember = randomMemberWithExclusion(Some(problemType), Nil).get //there should not be and empty list when we don't use exclusion
  def randomMember(): SimpleModelMember = randomMemberWithExclusion(None, Nil).get //there should not be and empty list when we don't use exclusion

  def randomMemberWithExclusion(problemType: ProblemType, exclude: Seq[SimpleModelMember]): Option[SimpleModelMember] = {
    randomMemberWithExclusion(Some(problemType), exclude)
  }

  def randomMemberWithExclusion(problemType: Option[ProblemType], exclude: Seq[SimpleModelMember]): Option[SimpleModelMember] = {

    val members = poolOfSimpleModelsByNames(poolOfSimpleModelsNames)
    val optionsToChooseFrom = members.filter(e => problemType.map(pt => e.canHandleProblemType(pt)).getOrElse(true)).diff(exclude)

    if (optionsToChooseFrom.nonEmpty)
      Some(optionsToChooseFrom.randElement)
    else None
  }

}

package com.automl.template.simple

import com.automl.evolution.dimension.hparameter.{HyperParametersField, HyperParametersGroup}
import com.automl.{ConfigProvider, PaddedLogging}
import com.automl.problemtype.ProblemType
import com.automl.template._
import com.automl.template.simple.perceptron.LinearPerceptron
import com.automl.teststrategy.TestStrategy
import com.typesafe.config.Config

import scala.collection.JavaConverters._

trait SimpleModelMember extends TemplateMember { self: PaddedLogging =>
  override def name: String = "simpleModel member"

  def canHandleProblemType: PartialFunction[ProblemType, Boolean]

  def modelKey: ModelKey = ???

  def testStrategy: TestStrategy = ???

  /**
    * Selects hpGroup based on the way how we decided to derive it. Either from coevolution or through regular mutation from TemplateTree
    */
  def getActiveHPGroup(config: Config, hpGroup: HyperParametersGroup[_], hyperParametersField: Option[HyperParametersField]) = {
    val hpCoevolutionIsEnabled = config.getBoolean("hyperParameterDimension.enabled")

    // It is assumed that we have only one relevant HPGroup per model
    def getRelevantHPGroup = hyperParametersField.get.modelsHParameterGroups.filter(_.isRelevantTo(this)).head

    if (hpCoevolutionIsEnabled) hpGroup else getRelevantHPGroup
  }
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

  val poolOfSimpleModels: Seq[SimpleModelMember] = Seq(/*DeepNeuralNetwork,*/ Bayesian, GLM, DT/*, SVM*//*,GradientBoosting,  KNN, LogisticRegression, SupportVectorRegression, LinearPerceptron*/)

  def poolOfSimpleModels(problemType: ProblemType): Seq[SimpleModelMember] = {
    poolOfSimpleModels.filter(_.canHandleProblemType(problemType))
  }

  val tdConfig = ConfigProvider.config.getConfig("evolution.templateDimension")
  lazy val poolOfSimpleModelsNames: Seq[String] = tdConfig.getStringList("poolOfSimpleModels").asScala

  // Maybe it is better to specify names inside models so that we don't need to pattern match here and suplicate stringg ids
  def poolOfSimpleModelsByNames(names: Seq[String]): Seq[SimpleModelMember] = names.map {
    case "logistic_regression" => LogisticRegression
    case "decision_tree" => DT
    case "bayesian" => Bayesian
  }

  def randomMember(problemType: ProblemType): SimpleModelMember = randomMemberWithExclusion(problemType, Nil).get //there should not be and empty list when we don't use exclusion

  def randomMemberWithExclusion(problemType: ProblemType, exclude: Seq[SimpleModelMember]): Option[SimpleModelMember] = {

    val optionsToChooseFrom = poolOfSimpleModelsByNames(poolOfSimpleModelsNames).filter(_.canHandleProblemType(problemType)).diff(exclude)
    if (optionsToChooseFrom.nonEmpty)
      Some(optionsToChooseFrom.randElement)
    else None
  }

}

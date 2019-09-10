package com.automl.evolution.evaluation

import akka.actor.ActorSystem
import com.automl.ConfigProvider
import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.dataset.Datasets
import com.automl.evolution.dimension.TemplateEvolutionDimension
import com.automl.evolution.dimension.hparameter.{DecisionTreeHPGroup, HyperParametersField, MaxDepth}
import com.automl.helper.FitnessResult
import com.automl.population.TPopulation
import com.automl.problemtype.ProblemType.MultiClassClassificationProblem
import com.automl.template.{LeafTemplate, NodeTemplate, TemplateMember, TemplateTree}
import com.automl.template.simple.DecisionTree
import com.typesafe.config.{Config, ConfigFactory}
import org.scalatest.{FunSuite, Matchers}

import scala.collection.mutable

class TemplateNSLCEvaluatorTest extends FunSuite with Matchers {

  test("Evaluation of individuals which only differs by their hyperparameters should be stored in a separate entries in a cache") {

    implicit val system = ActorSystem("AutoMLSuite-system")
    implicit val logPaddingSize = 0

    val testOverride: Config = ConfigFactory.parseString(
      """
        |evolution {
        |  hpGridSearch = false
        |  hyperParameterDimension {
        |     enabled = false
        |  }
        |}
      """.stripMargin)
    ConfigProvider.clearOverride.addOverride(testOverride)

    val decisionTree = LeafTemplate(DecisionTree())
    decisionTree.internalHyperParamsMap = Some(HyperParametersField(
      Seq(
        DecisionTreeHPGroup(Seq(MaxDepth(Some(2.0))))
      )
    ))

    val decisionTree2 = LeafTemplate(DecisionTree())
    decisionTree2.internalHyperParamsMap = Some(HyperParametersField(
      Seq(
        DecisionTreeHPGroup(Seq(MaxDepth(Some(3.0))))
      )
    ))

    val individuals = Seq(
      decisionTree,
      decisionTree2
    )

    val population = new TPopulation(individuals)
    val workingDF = Datasets.getGlassDataFrame(1234)

    val templateEvDim = new TemplateEvolutionDimension(initialPopulation = None, problemType = MultiClassClassificationProblem)

    val evaluator = new TemplateNSLCEvaluator(templateEvDim, None)

    implicit val implicitCache = mutable.Map[(TemplateTree[TemplateMember], Option[HyperParametersField], Long), FitnessResult]()

    evaluator.evaluateIndividuals(population, workingDF, MultiClassClassificationProblem, 1235)

    implicitCache.size shouldBe 2
  }

  //TODO fix the test
  ignore("Evaluation of individuals which are the same should not be stored in a separate entries in a cache") {

    implicit val system = ActorSystem("AutoMLSuite-system")
    implicit val logPaddingSize = 0

    val testOverride: Config = ConfigFactory.parseString(
      """
        |evolution {
        |  hpGridSearch = false
        |  hyperParameterDimension {
        |     enabled = false
        |  }
        |}
      """.stripMargin)
    ConfigProvider.clearOverride.addOverride(testOverride)

    val decisionTree = LeafTemplate(DecisionTree())
    decisionTree.internalHyperParamsMap = Some(HyperParametersField(
      Seq(
        DecisionTreeHPGroup(Seq(MaxDepth(Some(2.0))))
      )
    ))

    val decisionTree2 = LeafTemplate(DecisionTree())
    decisionTree2.internalHyperParamsMap = Some(HyperParametersField(
      Seq(
        DecisionTreeHPGroup(Seq(MaxDepth(Some(2.0))))
      )
    ))

    val individuals = Seq(
      decisionTree,
      decisionTree2
    )

    val population = new TPopulation(individuals)
    val workingDF = Datasets.getGlassDataFrame(1234)

    val templateEvDim = new TemplateEvolutionDimension(initialPopulation = None, problemType = MultiClassClassificationProblem)

    val evaluator = new TemplateNSLCEvaluator(templateEvDim, None)

    implicit val implicitCache = mutable.Map[(TemplateTree[TemplateMember], Option[HyperParametersField], Long), FitnessResult]()

    evaluator.evaluateIndividuals(population, workingDF, MultiClassClassificationProblem, 1235)

    implicitCache.size shouldBe 1
  }

}

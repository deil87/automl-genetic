package com.automl.evolution.evaluation

import akka.actor.ActorSystem
import com.automl.{ConfigProvider, EvaluatedTemplateData}
import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.dataset.Datasets
import com.automl.evolution.dimension.TemplateEvolutionDimension
import com.automl.evolution.dimension.hparameter._
import com.automl.evolution.diversity.MisclassificationDistance
import com.automl.population.TPopulation
import com.automl.problemtype.ProblemType.MultiClassClassificationProblem
import com.automl.template.{LeafTemplate, NodeTemplate, TemplateMember, TemplateTree}
import com.automl.template.ensemble.stacking.GenericStacking
import com.automl.template.simple.{Bayesian, DecisionTree, LogisticRegressionModel, RandomForest}
import org.scalatest.{FunSuite, Matchers}
import com.test.TestBase

class NeighboursFinderTest extends FunSuite with TestBase{

  implicit val system = ActorSystem("AutoMLSuite-system")
  implicit val logPaddingSize = 0

  test("consistency check is working") {

    ConfigProvider.clearOverride.addOverride(
      """
        |evolution {
        |  hyperParameterDimension {
        |    enabled = false
        |  }
        |  templateDimension {
        |    globalCVNumFolds = 3
        |  }
        |  hpGridSearch = false
        |}
        |consistencyCheck = true
      """)

    val individual1: TemplateTree[TemplateMember] =
      LeafTemplate(RandomForest(RandomForestHPGroup(Seq(MaxDepthRF(Some(6.0))))))

    val individual2: TemplateTree[TemplateMember] = LeafTemplate(Bayesian(BayesianHPGroup(Seq(Smoothing(Some(6.0))))))

    val individual3: TemplateTree[TemplateMember] =
      LeafTemplate(LogisticRegressionModel(LogisticRegressionHPGroup(Seq(RegParamLR(Some(0.2)), ElasticNet(Some(0.6))))))

    val originalPopulation = new TPopulation(Seq(individual1, individual2, individual3))

    val carData = Datasets.getCarDataFrame(1234).limit(50)

    val problemType = MultiClassClassificationProblem
    val t = new TemplateEvolutionDimension(initialPopulation = None, problemType = problemType)

    val evaluatedTD = t.evaluatePopulation(originalPopulation, carData, null)

    val neighboursFinder = new NeighboursFinder(new MisclassificationDistance)

    val data = evaluatedTD.randSubList
    neighboursFinder.findNeighbours(data, evaluatedTD.diff(data), 3, MultiClassClassificationProblem)

  }

  test("consistency check with ensembling individuals is working") {

    ConfigProvider.clearOverride.addOverride(
      """
        |evolution {
        |  hyperParameterDimension {
        |    enabled = false
        |  }
        |  templateDimension {
        |    globalCVNumFolds = 2
        |  }
        |  hpGridSearch = false
        |}
        |consistencyCheck = true
      """)

    val individual1: TemplateTree[TemplateMember] =
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(Bayesian(BayesianHPGroup(Seq(Smoothing(Some(7.0)))))),
          LeafTemplate(DecisionTree(DecisionTreeHPGroup(Seq(MaxDepth(Some(4.0)))), 1234)),
            LeafTemplate(LogisticRegressionModel(LogisticRegressionHPGroup(Seq(RegParamLR(Some(0.2)), ElasticNet(Some(0.7))))))
        )
      )

    val individual2: TemplateTree[TemplateMember] = LeafTemplate(Bayesian(BayesianHPGroup(Seq(Smoothing(Some(6.0))))))

    val individual3: TemplateTree[TemplateMember] =
      LeafTemplate(LogisticRegressionModel(LogisticRegressionHPGroup(Seq(RegParamLR(Some(0.2)), ElasticNet(Some(0.6))))))

    val originalPopulation = new TPopulation(Seq(individual1, individual2, individual3))

    val carData = Datasets.getCarDataFrame(1234).limit(50)

    val problemType = MultiClassClassificationProblem
    val t = new TemplateEvolutionDimension(initialPopulation = None, problemType = problemType)

    val evaluatedTD = t.evaluatePopulation(originalPopulation, carData, null)

    val neighboursFinder = new NeighboursFinder(new MisclassificationDistance)

    neighboursFinder.findNeighbours(evaluatedTD.take(2), evaluatedTD.drop(2), 3, MultiClassClassificationProblem)

  }
}

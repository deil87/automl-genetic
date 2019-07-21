package com.automl.evolution.mutation

import com.automl.ConfigProvider
import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.evolution.diversity.DistinctDiversityStrategy
import com.automl.helper.PopulationHelper
import com.automl.population.TPopulation
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.MultiClassClassificationProblem
import com.automl.template.ensemble.stacking.GenericStacking
import com.automl.template.simple.{Bayesian, DecisionTree, RandomForest}
import com.automl.template.{LeafTemplate, NodeTemplate, TemplateMember, TemplateTree}
import com.typesafe.config.{Config, ConfigFactory}
import org.scalatest.{FunSuite, Matchers}

class DepthDependentTemplateMutationStrategyTest extends FunSuite with Matchers{

  test("test that we can get desired depth of the tree during mutation") {

    val testOverride: Config = ConfigFactory.parseString(
      """
        |evolution {
        |  templateDimension {
        |    maxEnsembleDepth = 4
        |  }
        |}
      """.stripMargin)
    ConfigProvider.addOverride(testOverride)

    val distinctStrategy = new DistinctDiversityStrategy()
    val conf = ConfigProvider.config
    val mutationStrategy = new DepthDependentTemplateMutationStrategy(distinctStrategy, MultiClassClassificationProblem)(0)

    val individuals = Seq(
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(DecisionTree())
        )
      )
    )

    val population = new TPopulation(individuals)

    population.individuals.head.height shouldBe 2

    population.individuals.foreach(template => print(template.render))

    val newPopulation = mutationStrategy.mutateNTimes(population, 20)

    newPopulation.individuals.head.height should be <= 4

  }

  test("test that we don't get ensembling nodes with less then 2 child") {

    val individuals = Seq(
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(DecisionTree()),
          NodeTemplate(SparkGenericBagging(),
            Seq(
              LeafTemplate(DecisionTree()),
              NodeTemplate(SparkGenericBagging(),
                Seq(
                  LeafTemplate(DecisionTree())
                )
              )
            )
          )
        )
      )
    )

    val badPopulation = new TPopulation(individuals)

    val individuals2 = Seq(
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(DecisionTree()),
          NodeTemplate(SparkGenericBagging(),
            Seq(
              LeafTemplate(DecisionTree()),
              NodeTemplate(SparkGenericBagging(),
                Seq(
                  LeafTemplate(DecisionTree()),
                  LeafTemplate(Bayesian())  // <--- difference with `individuals` is only here
                )
              )
            )
          )
        )
      )
    )

    val correctPopulation = new TPopulation(individuals2)

    badPopulation.individuals.head.traverseAndCheck(_.subMembers.size >= 2) shouldBe false
    correctPopulation.individuals.head.traverseAndCheck(_.subMembers.size >= 2) shouldBe true
  }

  test("test that we don't get ensembling nodes with less then 2 child in real-world case") {
    val testOverride: Config = ConfigFactory.parseString(
      """
        |evolution {
        |  templateDimension {
        |    maxEnsembleDepth = 4
        |  }
        |}
      """.stripMargin)
    ConfigProvider.addOverride(testOverride)

    val distinctStrategy = new DistinctDiversityStrategy()
    val conf = ConfigProvider.config
    val mutationStrategy = new DepthDependentTemplateMutationStrategy(distinctStrategy, MultiClassClassificationProblem)(0)

    val individuals = Seq(
          LeafTemplate(DecisionTree())
    )

    val population = new TPopulation(individuals)


    val newPopulation = mutationStrategy.mutateNTimes(population, 20)

    newPopulation.individuals.foreach(template => println(template.render))
    newPopulation.individuals.head.traverseAndCheck(_.subMembers.size >= 2) shouldBe true
  }

  test("that we don't get into dead end as we compute level of mutation based on tree height") {

    //TODO
  }

  test("that we will receive new individual by applying mutateIndividual method(multi rerun setup)") {

    implicit val padding: Int = 0
    val strategy = new DepthDependentTemplateMutationStrategy(null, ProblemType.MultiClassClassificationProblem)

    val numberOfMutationIterations = 50
    (1 to numberOfMutationIterations).foreach{ _ =>
      val individual: NodeTemplate[TemplateMember] =
        NodeTemplate(SparkGenericBagging(),
          Seq(
            LeafTemplate(DecisionTree()),
            LeafTemplate(RandomForest()),
            LeafTemplate(Bayesian())
          )
        )

      val copyOfIndividual = ??? //individual
      val newIndividual = strategy.mutateIndividual(copyOfIndividual)

      println(individual.render)
      println(newIndividual.render)

      val areTheyEqual = individual == newIndividual
      areTheyEqual should be(false)
      individual shouldEqual individual
    }

  }

  test("that copy of LeafTree copy internal field") {
    val individual = LeafTemplate(DecisionTree())
    val copy = individual.copy()
//    individual == individual shouldBe true
//    individual == copy shouldBe false
  }

  test("that we will receive new unique population by using mutate method ") {

    implicit val padding: Int = 0
    val strategy = new DepthDependentTemplateMutationStrategy(null, ProblemType.MultiClassClassificationProblem)

    val individual1: TemplateTree[TemplateMember] =
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(DecisionTree()),
          LeafTemplate(RandomForest()),
          LeafTemplate(Bayesian())
        )
      )
    val individual2: TemplateTree[TemplateMember] =
      NodeTemplate(GenericStacking(),
        Seq(
          LeafTemplate(DecisionTree())
        )
      )
    val individual3: TemplateTree[TemplateMember] =
      LeafTemplate(DecisionTree())

    val originalPopulation = new TPopulation(Seq(individual1, individual2, individual3))

    originalPopulation.individuals diff Seq(individual3, individual1, individual2) shouldBe Seq()

    val newPopulation = strategy.mutate(originalPopulation)

    PopulationHelper.print(originalPopulation, "Original population:")
    PopulationHelper.print(newPopulation, "New population:")

    //Following way to test could not be used because of the scalatest's way of comparison and overriden equal method on TemplateTree
    //originalPopulation.individuals shouldNot contain theSameElementsAs(newPopulation.individuals)
    originalPopulation.individuals diff newPopulation.individuals shouldNot be( Seq())
  }

  test("that we will increase complexity") {
    implicit val padding: Int = 0

    val numberOfMutationPhases = 1000

    val strategy = new DepthDependentTemplateMutationStrategy(null, ProblemType.MultiClassClassificationProblem)
    val individual1: TemplateTree[TemplateMember] =
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(DecisionTree()),
          LeafTemplate(RandomForest()),
          LeafTemplate(Bayesian())
        )
      )
    val individual2: TemplateTree[TemplateMember] =
      NodeTemplate(GenericStacking(),
        Seq(
          LeafTemplate(DecisionTree())
        )
      )

    val individual3: TemplateTree[TemplateMember] =
      LeafTemplate(DecisionTree())

    val originalPopulation = new TPopulation(Seq(individual1, individual2, individual3))
    originalPopulation.individuals diff Seq(individual3, individual1, individual2) shouldBe Seq()

    val newPopulation = (1 to numberOfMutationPhases).foldLeft(originalPopulation) ((currentPopulation, index) => strategy.mutate(currentPopulation))

    PopulationHelper.print(originalPopulation, "Original population:")
    PopulationHelper.print(newPopulation, "New population:")

    newPopulation.depthComplexity shouldBe >= (originalPopulation.depthComplexity)
  }

}

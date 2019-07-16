package com.automl.evolution.mutation

import com.automl.ConfigProvider
import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.evolution.diversity.DistinctDiversityStrategy
import com.automl.population.TPopulation
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.MultiClassClassificationProblem
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

  test("that we will receive new individual") {

    implicit val padding: Int = 0
    val strategy = new DepthDependentTemplateMutationStrategy(null, ProblemType.MultiClassClassificationProblem)

    val individual: TemplateTree[TemplateMember] =
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(DecisionTree()),
          LeafTemplate(RandomForest()),
          LeafTemplate(Bayesian())
        )
      )

    val newIndividual = strategy.mutateIndividual(individual)

    println(individual.render)
    println(newIndividual.render)

    individual should not equal newIndividual
    individual shouldEqual individual
  }

}

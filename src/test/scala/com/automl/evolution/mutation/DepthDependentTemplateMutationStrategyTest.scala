package com.automl.evolution.mutation

import com.automl.{ConfigProvider, TPopulation}
import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.evolution.diversity.DistinctDiversityStrategy
import com.automl.problemtype.ProblemType.MultiClassClassificationProblem
import com.automl.template.{LeafTemplate, NodeTemplate}
import com.automl.template.simple.{Bayesian, DecisionTree, LogisticRegressionModel}
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


    def mutationHelper(population: TPopulation, countdown: Int): TPopulation = {
      if(countdown == 0) population
      else {
        val mutatedPopulation = mutationStrategy.mutate(population)
        mutatedPopulation.individuals.foreach(template => print(template.render))
        mutationHelper(mutatedPopulation, countdown - 1)
      }
    }

    val newPopulation = mutationHelper(population, 20)

    newPopulation.individuals.head.height should be <= 4

  }

  test("test that we don't get ensembling nodes with less then 2 child") {

    val distinctStrategy = new DistinctDiversityStrategy()
    val mutationStrategy = new DepthDependentTemplateMutationStrategy(distinctStrategy, MultiClassClassificationProblem)(0)

    val individuals = Seq(
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(DecisionTree())
        )
      )
    )

    val population = new TPopulation(individuals)

    def mutationHelper(population: TPopulation, countdown: Int): TPopulation = {
      if(countdown == 0) population
      else {
        val mutatedPopulation = mutationStrategy.mutate(population)
        mutationHelper(mutatedPopulation, countdown - 1)
      }
    }

    val newPopulation = mutationHelper(population, 10)

    newPopulation.individuals.foreach(template => print(template.render))
    newPopulation.individuals.head.height should be >= 3

  }

}

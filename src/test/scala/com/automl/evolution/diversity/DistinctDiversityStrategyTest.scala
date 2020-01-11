package com.automl.evolution.diversity

import com.automl.helper.PopulationHelper
import com.automl.population.{GenericPopulationBuilder, TPopulation}
import com.automl.template.simple._
import com.automl.template.{LeafTemplate, TemplateMember, TemplateTree, TestGit}
import org.scalatest.{Matchers, WordSpec}

class DistinctDiversityStrategyTest extends WordSpec with Matchers{

  "DistinctDiversityStrategyTest" should {

    "result in a diverse population" ignore {
      val distinctStrategy = new DistinctDiversityStrategy()

      val seed: Seq[LeafTemplate[SimpleModelMember]] = Seq(
        LeafTemplate(Bayesian())
      )

      val poolOfSimpleModels: Seq[SimpleModelMember] = {
        import SimpleModelMember._
        Seq(DeepNeuralNetwork, NeuralNetwork, Bayesian, GLM, DT, KNN)
      }

      val seedPopulation = new TPopulation(seed)

      val population = GenericPopulationBuilder.fromSeedPopulation(seedPopulation).withSize(6).build

      def getRandomEnsemblingMember(input: TemplateTree[TemplateMember]) = LeafTemplate(poolOfSimpleModels.randElement)

      val diversePopulation = distinctStrategy.apply(population, getRandomEnsemblingMember)
      PopulationHelper.print(diversePopulation)

      diversePopulation.size shouldBe population.size
    }

    "TestGit should work" ignore {
        new TestGit().receiveValue shouldBe 42
    }

  }
}

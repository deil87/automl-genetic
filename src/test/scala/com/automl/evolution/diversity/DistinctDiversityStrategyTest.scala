package com.automl.evolution.diversity

import com.automl.TPopulation
import com.automl.helper.PopulationHelper
import com.automl.template.simple._
import com.automl.template.{LeafTemplate, TemplateMember, TemplateTree}
import org.scalatest.{Matchers, WordSpec}

class DistinctDiversityStrategyTest extends WordSpec with Matchers{

  "DistinctDiversityStrategyTest" should {

    "result in a diverse population" in {
      val distinctStrategy = new DistinctDiversityStrategy()

      val seed: Seq[LeafTemplate[SimpleModelMember]] = Seq(
        LeafTemplate(Bayesian())
      )

      val poolOfSimpleModels: Seq[SimpleModelMember] = {
        import SimpleModelMember._
        Seq(DeepNeuralNetwork, NeuralNetwork, Bayesian, GLM, DT, KNN)
      }

      val seedPopulation = new TPopulation(seed)

      val population = TPopulation.fromSeedPopulation(seedPopulation).withSize(6).build

      def getRandomEnsemblingMember(input: TemplateTree[TemplateMember]) = LeafTemplate(poolOfSimpleModels.randElement)

      (population.individuals.distinct.size < population.size) shouldBe true

      val diversePopulation = distinctStrategy.apply(population, getRandomEnsemblingMember)
      PopulationHelper.print(diversePopulation)

      diversePopulation.size shouldBe population.size
    }

  }
}

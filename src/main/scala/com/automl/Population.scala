package com.automl

import com.automl.evolution.mutation.MutationProbabilities
import com.automl.template.{TemplateMember, TemplateTree}

import scala.util.Random

class Population(val individuals: Seq[ TemplateTree[TemplateMember]], val mutationProbabilities: MutationProbabilities = null) {

  def size: Int = individuals.length
}

object Population {
  lazy val firstEverPopulation = new Population(TemplateTree.firstPopulation.toSeq)

  case class PopulationBuilder(individuals: Seq[ TemplateTree[TemplateMember]], mutationProbabilities: MutationProbabilities) {
    def withSize(populationSize: Int): PopulationBuilder = {
      val rnd = new Random()
      val spanned = Seq.fill(populationSize)(individuals(rnd.nextInt(individuals.size)))
      this.copy(individuals = spanned)
    }

    def withDefaultMutationProbs: PopulationBuilder = {
      this.copy(mutationProbabilities = MutationProbabilities())
    }

    def build: Population = new Population(individuals, mutationProbabilities)
  }

  def fromSeedPopulation(classifiersPopulation: Population): PopulationBuilder = {
    PopulationBuilder(classifiersPopulation.individuals, null)
  }
}

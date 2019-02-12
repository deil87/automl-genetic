package com.automl

import com.automl.evolution.mutation.MutationProbabilities
import com.automl.template.{TemplateMember, TemplateTree}

import scala.util.Random

trait Population[T] {
  def individuals: Seq[T]

  def size: Int = individuals.length

  def nonEmpty: Boolean = size != 0
}

class TPopulation(val individuals: Seq[ TemplateTree[TemplateMember]],
                  val mutationProbabilities: MutationProbabilities = null) extends Population[TemplateTree[TemplateMember]]{
}

object TPopulation {

  case class PopulationBuilder(individuals: Seq[ TemplateTree[TemplateMember]], mutationProbabilities: MutationProbabilities) {
    def withSize(populationSize: Int): PopulationBuilder = {
      if(populationSize >= individuals.size) {
        val rnd = new Random()
        val diff = populationSize - individuals.size
        val spanned = Seq.fill(diff)(individuals(rnd.nextInt(individuals.size)))
        this.copy(individuals = individuals ++ spanned)
      } else {
        val rnd = new Random()
        val shrinked = Seq.fill(populationSize)(individuals(rnd.nextInt(individuals.size)))
        this.copy(individuals = shrinked)
      }
    }

    def withDefaultMutationProbs: PopulationBuilder = {
      this.copy(mutationProbabilities = MutationProbabilities())
    }

    def build: TPopulation = new TPopulation(individuals, mutationProbabilities)
  }

  def fromSeedPopulation(classifiersPopulation: TPopulation): PopulationBuilder = {
    PopulationBuilder(classifiersPopulation.individuals, null)
  }
}

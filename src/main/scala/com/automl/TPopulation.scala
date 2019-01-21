package com.automl

import com.automl.evolution.mutation.MutationProbabilities
import com.automl.template.{TemplateMember, TemplateTree}

import scala.util.Random

trait Population[T] {
  def individuals: Seq[T]

  def size: Int = individuals.length
}

class TPopulation(val individuals: Seq[ TemplateTree[TemplateMember]],
                  val mutationProbabilities: MutationProbabilities = null) extends Population[TemplateTree[TemplateMember]]{
}

object TPopulation {
  lazy val firstEverPopulation = new TPopulation(TemplateTree.firstPopulation.toSeq)

  case class PopulationBuilder(individuals: Seq[ TemplateTree[TemplateMember]], mutationProbabilities: MutationProbabilities) {
    def withSize(populationSize: Int): PopulationBuilder = {
      val rnd = new Random()
      val spanned = Seq.fill(populationSize)(individuals(rnd.nextInt(individuals.size)))
      this.copy(individuals = spanned)
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

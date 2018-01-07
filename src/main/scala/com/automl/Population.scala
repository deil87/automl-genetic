package com.automl

import com.automl.template.{TemplateMember, TemplateTree}

import scala.util.Random

class Population(val individuals: Seq[ TemplateTree[TemplateMember]]) {



}

object Population {
  lazy val firstEverPopulation = TemplateTree.firstPopulation

  case class PopulationBuilder(individuals: Seq[ TemplateTree[TemplateMember]]) {
    def withSize(populationSize: Int): PopulationBuilder = {
      val rnd = new Random()
      val spanned = Seq.fill(populationSize)(individuals(rnd.nextInt(individuals.size))).toSeq
      this.copy(individuals = spanned)
    }

    def build: Population = new Population(individuals)
  }

  def fromSeed(classifiers: Seq[ TemplateTree[TemplateMember]]): PopulationBuilder = {
    PopulationBuilder(classifiers)
  }
}

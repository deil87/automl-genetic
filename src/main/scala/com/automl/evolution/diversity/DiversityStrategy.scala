package com.automl.evolution.diversity

import com.automl.Population
import com.automl.template.{TemplateMember, TemplateTree}

trait DiversityStrategy {
  def apply(input: Population, fun: TemplateTree[TemplateMember] => TemplateTree[TemplateMember]): Population
}

class DistinctDiversityStrategy extends DiversityStrategy {
  def apply(input: Population, fun: TemplateTree[TemplateMember] => TemplateTree[TemplateMember]): Population = {
    val populationSize = input.individuals.length
    var distinctTemplates: Seq[TemplateTree[TemplateMember]] = input.individuals map fun distinct

    def mutate(numberOfNeededMutants: Int): Seq[TemplateTree[TemplateMember]] = {
      val newMutants = scala.util.Random.shuffle(distinctTemplates).take(numberOfNeededMutants) map fun

      distinctTemplates = (distinctTemplates ++ newMutants).distinct
      if(distinctTemplates.length != populationSize )
        mutate(populationSize - distinctTemplates.length)
      else
        distinctTemplates
    }

    mutate(populationSize - distinctTemplates.length)

    new Population(distinctTemplates)
  }
}
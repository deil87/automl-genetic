package com.automl.population

import com.automl.helper.PopulationHelper
import com.automl.template.{TemplateMember, TemplateTree}

class TPopulation(val individuals: Seq[ TemplateTree[TemplateMember]])
    extends Population[TemplateTree[TemplateMember]]{

  def depthComplexity: Int = individuals.foldLeft(0)((res, next) => res + next.height)

  override def render: Unit = PopulationHelper.print(this, "Rendered population")
}

package com.automl.helper

import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.{Evaluated, EvaluatedTemplateData, TPopulation}
import com.automl.template.{TemplateMember, TemplateTree}
import com.typesafe.scalalogging.LazyLogging

object PopulationHelper extends LazyLogging{
  def print(population: TPopulation, prefix: String = ""): Unit = {
    val prefixToPrepend = if (prefix.nonEmpty) prefix + " : " else prefix
    logger.info(prefixToPrepend  + renderIndividuals(population.individuals))
  }

  def renderIndividuals(individuals: Seq[TemplateTree[TemplateMember]]): String = {
    individuals.zipWithIndex.map { case (individual, idx) => f"$idx ) ${TemplateTreeHelper.renderAsString_v2(individual)}" }.mkString("\n\t\t\t\t\t", "\n\t\t\t\t\t", "")
  }
  /*def renderEvaluatedIndividuals(individuals: Seq[EvaluatedTemplateData]): String = {
    individuals.map(evd => (evd.fitness.getCorrespondingMetric, evd.template)).sortBy(_._1).map { case ( correspondingMetric, template) => f" - ${TemplateTreeHelper.renderAsString_v2(template)} $correspondingMetric" }.mkString("\n\t\t\t\t\t", "\n\t\t\t\t\t", "")
  }*/
  def renderEvaluatedIndividuals[T <: Evaluated[T]](individuals: Seq[T]): String = {
    individuals.sortWith((a,b) => a.compare(b)).map(evd => (evd.result, evd.item))

      .map {
        case ( correspondingMetric, item:HyperParametersField) => f" - ${renderHyperParameterField(item)} $correspondingMetric"
        case ( correspondingMetric, template:TemplateTree[TemplateMember] ) => f" - ${TemplateTreeHelper.renderAsString_v2(template)} $correspondingMetric"
      }
      .mkString("\n\t\t\t\t\t", "\n\t\t\t\t\t", "")
  }

  def renderHyperParameterField(field: HyperParametersField): String = {
    field.modelsHParameterGroups.map(group => group.hpParameters.map(parameter => s"$parameter").mkString(" , ")).mkString(" | ")
  }
}

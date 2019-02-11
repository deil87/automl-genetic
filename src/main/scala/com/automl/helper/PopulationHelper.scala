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

  //TODO move to EvolutionDimension or a trait that will be mixed to EvolutionDimension
  def renderEvaluatedIndividuals[T <: Evaluated[T]](individuals: Seq[T]): String = {
    individuals
      .sortWith((a,b) => a.compare(b)).map(evd => (evd.result, evd.item, evd.params))
      .map {
        case ( correspondingMetric, item:HyperParametersField, _) => f" - $item $correspondingMetric"
        case ( correspondingMetric, template:TemplateTree[TemplateMember], params: HyperParametersField) => f" - ${TemplateTreeHelper.renderAsString_v2(template)} \n Evaluation: $correspondingMetric hp: $params"
        case _ => throw new IllegalStateException("Unmanaged cases")
      }
      .mkString("\n\t\t\t\t\t", "\n\t\t\t\t\t", "")
  }

}

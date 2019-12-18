package com.automl.helper

import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.population.TPopulation
import com.automl.{Evaluated, EvaluatedTemplateData, PaddedLogging}
import com.automl.template.{TemplateMember, TemplateTree}
import com.typesafe.scalalogging.LazyLogging

object PopulationHelper extends PaddedLogging{


  override def logPaddingSize: Int = 0

  def print(population: TPopulation, prefix: String = ""): Unit = {
    val prefixToPrepend = if (prefix.nonEmpty) prefix + " : " else prefix
    info(prefixToPrepend  + renderIndividuals(population.individuals))
  }

  def renderIndividuals(individuals: Seq[TemplateTree[TemplateMember]]): String = {
    individuals.zipWithIndex.map { case (individual, idx) => f"$idx ) ${TemplateTreeHelper.renderAsString_v2(individual)}" }.mkString("\n\t\t\t\t\t", "\n\t\t\t\t\t", "")
  }

  //TODO move to EvolutionDimension or a trait that will be mixed to EvolutionDimension
  def renderEvaluatedIndividuals[T <: Evaluated[T]](individuals: Seq[T], prefix: String = ""): String = {
    val prefixToPrepend = if (prefix.nonEmpty) prefix + " : \n" else prefix
    prefixToPrepend  + individuals
      .sortWith((a,b) => a.compare(b) > 0)
      .map(evd => (evd.idShort, evd.result, evd.item, evd.params))
      .map {
        case (idShort, correspondingMetric, item:HyperParametersField, _) => f" - $item $correspondingMetric"
        case (idShort, correspondingMetric, template:TemplateTree[TemplateMember], _) =>
          f" - ${TemplateTreeHelper.renderAsString_v2(template)} $idShort \t\t\t Evaluation: $correspondingMetric"
        case _ => throw new IllegalStateException("Unmanaged cases")
      }
      .mkString("\n\t\t\t\t\t", "\n\t\t\t\t\t", "")
  }

}

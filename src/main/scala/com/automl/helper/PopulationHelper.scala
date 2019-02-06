package com.automl.helper

import com.automl.TPopulation
import com.typesafe.scalalogging.LazyLogging

object PopulationHelper extends LazyLogging{
  def print(population: TPopulation, prefix: String = ""): Unit = {
    val prefixToPrepend = if (prefix.nonEmpty) prefix + " : " else prefix
    logger.info(prefixToPrepend  + population.individuals.zipWithIndex.map { case (individual, idx) => f"$idx ) ${TemplateTreeHelper.renderAsString_v2(individual)}" }.mkString("\n\t\t\t\t\t", "\n\t\t\t\t\t", ""))
  }
}

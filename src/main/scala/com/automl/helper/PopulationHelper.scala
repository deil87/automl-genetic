package com.automl.helper

import com.automl.Population
import com.typesafe.scalalogging.LazyLogging

object PopulationHelper extends LazyLogging{
  def print(population: Population): Unit =
    population.individuals.zipWithIndex.foreach{case (individual, idx) =>
      logger.info(idx + ") " + TemplateTreeHelper.renderAsString_v2(individual) + "\n")
    }

}

package com.automl.helper

import com.automl.Population
import com.automl.template.{TemplateMember, TemplateTree}

object PopulationHelper {
  def print(population: Population): Unit =
    population.individuals.zipWithIndex.foreach{case (individual, idx) => println(idx + ") " + TemplateTreeHelper.print2(individual) + "\n")}

}

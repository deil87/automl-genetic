package com.automl.report

import com.automl.EvaluatedTemplateData
import com.automl.helper.TemplateTreeHelper

object AutoMLReporter {

  def show(bestIndividuals: collection.mutable.PriorityQueue[EvaluatedTemplateData]): Unit = {
    // Final evaluation on different test data. Consider winners from all evolutions(evolutionNumbers) but put more faith into last ones because they have been chosen based on results on bigger  validation sets(better representative of a population).
    if(bestIndividuals.isEmpty) {
      println("\n##############################################################")
      println("None of the evolutions were finished within given time constraint")
    } else {

      val winner = bestIndividuals.dequeue() // TODO warning - could be empty

      println("\n##############################################################")
      println("Fitness value of the BEST template: " +  winner.fitness.fitnessError)
      println("Best template: " + TemplateTreeHelper.print2(winner.template)) // TODO make print2 actually a printing method
      println("Other best individuals results:\n" + bestIndividuals.dequeueAll.map(_.fitness.fitnessError).mkString(",\n"))

    }
  }
}
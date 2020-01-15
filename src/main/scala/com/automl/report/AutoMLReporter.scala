package com.automl.report

import com.automl.EvaluatedTemplateData
import com.automl.helper.{FitnessResult, TemplateTreeHelper}
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}

object AutoMLReporter {

  def show(bestIndividuals: Iterator[EvaluatedTemplateData], problemType: ProblemType): Unit = {
    // Final evaluation on different test data. Consider winners from all evolutions(evolutionNumbers) but put more faith into last ones because they have been chosen based on results on bigger  validation sets(better representative of a population).
    if(bestIndividuals.isEmpty) {
      println("\n########################## AutoML results #################################################")
      println("None of the evolutions were finished within given time constraint")
    } else {

      val winner = bestIndividuals.next()

      println("\n########################## AutoML results #################################################")
      println("Best template: \n" + winner.render(problemType))
      println("\nOther best individuals results:\n" + bestIndividuals.map{etd => etd.render(problemType) }.mkString(",\n\n")
      )
    }
  }
}

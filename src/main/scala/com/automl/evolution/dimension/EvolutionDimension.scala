package com.automl.evolution.dimension

import com.automl.problemtype.ProblemType
import com.automl.{EvaluatedTemplateData, Population}
import org.apache.spark.ml.param.Params
import org.apache.spark.sql.DataFrame


/**
  * For fast implementions of coevolutions
  *
  * @tparam T
  */
//TODO problemType: ProblemType parameter might be moved somewhere to a field like `population`
trait EvolutionDimension[PopulationType <: Population[T], T, EvaluatedResult] {

  var _population: PopulationType

  // Template pattern
  def evolve(population: PopulationType, workingDF: DataFrame): PopulationType = {
    val evaluatedOriginalPopulation = evaluatePopulation(population, workingDF)

    val selectedParents= selectParents(evaluatedOriginalPopulation)

    val selectedParentsPopulation = extractIndividualsFromEvaluatedIndividuals(selectedParents)

    val offspring = mutateParentPopulation(selectedParentsPopulation)

    val evaluationsForOffspring = evaluatePopulation(offspring, workingDF)

    val survivedForNextGenerationEvaluatedTemplates = selectSurvived(population.size, evaluationsForOffspring)

    extractIndividualsFromEvaluatedIndividuals(survivedForNextGenerationEvaluatedTemplates)
  }

  def selectParents(evaluatedIndividuals: Seq[EvaluatedResult]):  Seq[EvaluatedResult]

  // or maybe we can return parents without scores? But this method could be useful in other cases
  def extractIndividualsFromEvaluatedIndividuals(evaluatedIndividuals: Seq[EvaluatedResult]): PopulationType

  //Consider renaming mutateParents
  def mutateParentPopulation(population: PopulationType): PopulationType

  //meaning each individual separately and not as a whole
  def evaluatePopulation(population: PopulationType, workingDF: DataFrame): Seq[EvaluatedResult]

  def selectSurvived(populationSize:Int, evaluatedIndividuals: Seq[EvaluatedResult]):  Seq[EvaluatedResult]

  def getPopulation: PopulationType = _population

  def getBestFromPopulation(workingDF: DataFrame): EvaluatedResult

}





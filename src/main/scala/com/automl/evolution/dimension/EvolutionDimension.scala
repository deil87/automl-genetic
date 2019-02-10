package com.automl.evolution.dimension

import com.automl.{Evaluated, EvaluatedTemplateData, Population}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.sql.DataFrame

import scala.collection.mutable


/**
  * For fast implementions of coevolutions
  *
  * @tparam T
  */
//TODO problemType: ProblemType parameter might be moved somewhere to a field like `population`
trait EvolutionDimension[PopulationType <: Population[T], T, EvaluatedResult <: Evaluated[EvaluatedResult]] extends LazyLogging{

  var _population: PopulationType

  var _evaluatedPopulation: Seq[EvaluatedResult] = Nil

  //We should have best population and bestIndividual as a separate things
  def getInitialPopulation: PopulationType = {
    val useMetaDB = false
    if(useMetaDB) getInitialPopulationFromMetaDB
    else getInitialColdStartPopulation
  }

  def getInitialPopulationFromMetaDB: PopulationType

  def getInitialColdStartPopulation: PopulationType

  def evolutionDimensionLabel: String

  implicit val individualsEvaluationCache:mutable.Map[(T, Long), EvaluatedResult#FitnessType] //cache store anyone regardless of performance and could be/should be capped to prevent OOM

  val hallOfFame: mutable.PriorityQueue[EvaluatedResult]

  // Template pattern
  def evolveFromLastPopulation(workingDF: DataFrame): PopulationType = {
    evolve(getPopulation, workingDF)
  }

  def evolve(population: PopulationType, workingDF: DataFrame): PopulationType = {
    showCurrentPopulation()
    logger.debug("Starting next evolution...")
    val evaluatedOriginalPopulation = evaluatePopulation(population, workingDF)

    logger.debug("Selecting parents:")
    val selectedParents= selectParents(evaluatedOriginalPopulation)

    val selectedParentsPopulation = extractIndividualsFromEvaluatedIndividuals(selectedParents)

    logger.debug("Mutating parents:")
    val offspring = mutateParentPopulation(selectedParentsPopulation)

    logger.debug("Evaluating offspring:")
    val evaluatedOffspring = evaluatePopulation(offspring, workingDF)

    updateHallOfFame(evaluatedOffspring)

    logger.debug("Selecting survivals:")
    val survivedForNextGenerationEvaluatedTemplates = selectSurvived(population.size, evaluatedOffspring)

    _evaluatedPopulation = survivedForNextGenerationEvaluatedTemplates

    val evolvedNewGeneration = extractIndividualsFromEvaluatedIndividuals(survivedForNextGenerationEvaluatedTemplates)
    logger.debug("Evolution is finished.")

    _population = evolvedNewGeneration
    _population
  }

  def updateHallOfFame(evaluatedIndividuals: Seq[EvaluatedResult]):Unit

  def showCurrentPopulation(): Unit

  def getBestFromHallOfFame:T

  def selectParents(evaluatedIndividuals: Seq[EvaluatedResult]):  Seq[EvaluatedResult]

  // or maybe we can return parents without scores? But this method could be useful in other cases
  def extractIndividualsFromEvaluatedIndividuals(evaluatedIndividuals: Seq[EvaluatedResult]): PopulationType

  //Consider renaming mutateParents
  def mutateParentPopulation(population: PopulationType): PopulationType

  //meaning each individual separately and not as a whole
  def evaluatePopulation(population: PopulationType, workingDF: DataFrame): Seq[EvaluatedResult]

  def selectSurvived(populationSize:Int, evaluatedIndividuals: Seq[EvaluatedResult]):  Seq[EvaluatedResult]

  def getPopulation: PopulationType = if(_population.nonEmpty) _population else getInitialPopulation

  def getEvaluatedPopulation: Seq[EvaluatedResult] = _evaluatedPopulation

  def getBestFromPopulation(workingDF: DataFrame): EvaluatedResult

}





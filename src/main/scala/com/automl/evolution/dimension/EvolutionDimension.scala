package com.automl.evolution.dimension

import com.automl.helper.PopulationHelper
import com.automl.population.{Population, TPopulation}
import com.automl.{Evaluated, EvaluatedTemplateData, PaddedLogging}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.sql.DataFrame

import scala.collection.mutable


/**
  * For fast implementions of coevolutions
  *
  * @tparam T
  */
//TODO problemType: ProblemType parameter might be moved somewhere to a field like `population`
trait EvolutionDimension[PopulationType <: Population[T], T, EvaluatedResult <: Evaluated[EvaluatedResult]] { self: PaddedLogging =>

  def dimensionName: String

  var _population: PopulationType

  var _evaluatedPopulation: Seq[EvaluatedResult] = Nil

  var currentWorkingDFSize:Long = 0

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
    debug(s"Starting next $dimensionName evolution...")
    val evaluatedOriginalPopulation = getLastEvaluatedPopulation(workingDF)

    debug("Selecting parents:")
    val selectedParents = selectParents(evaluatedOriginalPopulation)

    val selectedParentsPopulation = extractIndividualsFromEvaluatedIndividuals(selectedParents)

    debug("Mutating parents:")
    val offspring = mutateParentPopulation(selectedParentsPopulation, getPopulation)

    debug("Evaluating offspring:")
    val evaluatedOffspring = evaluatePopulation(offspring, workingDF)

    debug("Updating hallOfFame:")
    updateHallOfFame(evaluatedOffspring)

    debug("Selecting survivals:")
    val survivedForNextGenerationEvaluatedTemplates = selectSurvived(population.size, evaluatedOffspring)

    _evaluatedPopulation = survivedForNextGenerationEvaluatedTemplates

    val evolvedNewGeneration: PopulationType = extractIndividualsFromEvaluatedIndividuals(survivedForNextGenerationEvaluatedTemplates)
    evolvedNewGeneration.render
    debug("Evolution is finished.") // TODO we should not stop evolution after one run of generation.

    _population = evolvedNewGeneration
    _population
  }

  def updateHallOfFame(evaluatedIndividuals: Seq[EvaluatedResult]):Unit = {
    val hallOfFameUpdateSize = 5  // TODO Config
    hallOfFame.headOption.map{bestAtAllTimes =>
      //TODO >= should be <= when we have "the less the better" approach
      hallOfFame ++= evaluatedIndividuals.filter(_.compare(bestAtAllTimes)).take(hallOfFameUpdateSize)
    }.getOrElse{
      hallOfFame ++= evaluatedIndividuals.take(hallOfFameUpdateSize)
    }
  }

  def showCurrentPopulation(): Unit

  def getBestFromHallOfFame:T

  def selectParents(evaluatedIndividuals: Seq[EvaluatedResult]):  Seq[EvaluatedResult]

  // or maybe we can return parents without scores? But this method could be useful in other cases
  def extractIndividualsFromEvaluatedIndividuals(evaluatedIndividuals: Seq[EvaluatedResult]): PopulationType

  //Consider renaming mutateParents
  def mutateParentPopulation(population: PopulationType, notToIntersectWith: PopulationType): PopulationType

  //meaning each individual separately and not as a whole
  def evaluatePopulation(population: PopulationType, workingDF: DataFrame): Seq[EvaluatedResult]

  def selectSurvived(populationSize:Int, evaluatedIndividuals: Seq[EvaluatedResult]):  Seq[EvaluatedResult]

  def getPopulation: PopulationType = if(_population.nonEmpty) _population else getInitialPopulation

  def getEvaluatedPopulation: Seq[EvaluatedResult] = _evaluatedPopulation

  def getLastEvaluatedPopulation(workingDF: DataFrame): Seq[EvaluatedResult] = {
    val newWorkingDFSize = workingDF.count()
    val evaluated = if (getEvaluatedPopulation.nonEmpty && newWorkingDFSize == currentWorkingDFSize) {
      debug(s"Taking $dimensionName evaluated population from previous generation.")
      getEvaluatedPopulation
    } else {
      if (newWorkingDFSize != currentWorkingDFSize && currentWorkingDFSize != 0) {
        debug(s"Reevaluating $dimensionName population due to increased working dataset size from $currentWorkingDFSize to $newWorkingDFSize")
        currentWorkingDFSize = newWorkingDFSize
        evaluatePopulation(getPopulation, workingDF)
      } else {
        currentWorkingDFSize = newWorkingDFSize
        debug(s"Evaluating $dimensionName population for the very first time.")
        evaluatePopulation(getPopulation, workingDF)
      }
    }
    evaluated
  }

  def getBestFromPopulation(workingDF: DataFrame): EvaluatedResult

}





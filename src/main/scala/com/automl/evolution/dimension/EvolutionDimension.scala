package com.automl.evolution.dimension

import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.evolution.evaluation.EvaluationContextInfo
import com.automl.helper.{FitnessResult, PopulationHelper}
import com.automl.population.{Population, TPopulation}
import com.automl.template.{TemplateMember, TemplateTree}
import com.automl.{Evaluated, EvaluatedTemplateData, EvaluationRules, PaddedLogging}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.sql.DataFrame

import scala.collection.mutable


/**
  * For fast implementations of co-evolutions
  *
  * @tparam T
  */
//TODO problemType: ProblemType parameter might be moved somewhere to a field like `population`
trait EvolutionDimension[PopulationType <: Population[T], T, EvaluatedResult <: Evaluated[EvaluatedResult]] extends EvaluationRules { self: PaddedLogging =>

  def dimensionName: String

  var _population: PopulationType

  var _evaluatedEvolvedPopulation: Seq[EvaluatedResult] = Nil

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

  //Depends on HyperParametersField which should be generic type as well
//  implicit val individualsEvaluationCacheExtended: mutable.Map[(T, Option[HyperParametersField], Long), EvaluatedResult#FitnessType]

  val hallOfFame: mutable.PriorityQueue[EvaluatedResult]

  // Template pattern
  def evolveFromLastPopulation(workingDF: DataFrame, evaluationContext: EvaluationContextInfo): PopulationType = {
    evolve(getPopulation, workingDF, evaluationContext)
  }

  def evolve(population: PopulationType, workingDF: DataFrame, evaluationContextInfo: EvaluationContextInfo): PopulationType = {
    showCurrentPopulation()
    debug(s"Starting next $dimensionName evolution...")
    val evaluatedOriginalPopulation = getLastEvaluatedPopulation(workingDF, evaluationContextInfo)
    evaluatedOriginalPopulation.foreach(eo => eo.setEvaluationContextInfo(evaluationContextInfo))

    debug("Selecting parents:")
    val selectedParents = selectParents(evaluatedOriginalPopulation)

    val selectedParentsPopulation = extractIndividualsFromEvaluatedIndividuals(selectedParents)

    debug("Mutating parents:")
    val offspring = mutateParentPopulation(selectedParentsPopulation, getPopulation)

    debug("Evaluating offspring:")
    val evaluatedOffspring: Seq[EvaluatedResult] = evaluatePopulation(offspring, workingDF, evaluationContextInfo)

    debug("Updating hallOfFame:")
    updateHallOfFame(evaluatedOffspring)

    debug("Selecting survivals:")
    val survivedForNextGenerationEvaluatedTemplates = selectSurvived(population.size, evaluatedOffspring)

    _evaluatedEvolvedPopulation = survivedForNextGenerationEvaluatedTemplates

    val evolvedNewGeneration: PopulationType = extractIndividualsFromEvaluatedIndividuals(survivedForNextGenerationEvaluatedTemplates)
    evolvedNewGeneration.render
    debug("Evolution is finished.") // TODO we should not stop evolution after one run of generation.

    _population = evolvedNewGeneration
    _population
  }

  def updateHallOfFame(evaluatedIndividuals: Seq[EvaluatedResult]):Unit = {
    val hallOfFameUpdateSize = 5  // TODO Config
    hallOfFame.headOption.map{bestAtAllTimes =>
      debug("WARNING!!! Check that updateHallOfFame supports regressiong problem")
      val goingToTheHallOfFame = evaluatedIndividuals.toSet
        .filter(r => r.betterThanOrEqual(bestAtAllTimes) > 0)
        .take(hallOfFameUpdateSize)
        .diff(hallOfFame.toSet)
      info(s"Following templates were added to the hall of fame: ${goingToTheHallOfFame.map(_.idShort).mkString(",")}")
      hallOfFame ++= goingToTheHallOfFame
    }.getOrElse{
      hallOfFame ++= evaluatedIndividuals.sorted.reverse.take(hallOfFameUpdateSize) // make sure we don't sort unnecessary second time
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
  def evaluatePopulation(population: PopulationType, workingDF: DataFrame, evaluationContextInfo: EvaluationContextInfo): Seq[EvaluatedResult]

  def selectSurvived(populationSize:Int, evaluatedIndividuals: Seq[EvaluatedResult]):  Seq[EvaluatedResult]

  def getPopulation: PopulationType = if(_population.nonEmpty) _population else getInitialPopulation

  // TODO do we it if we have getLastEvaluatedPopulation ?
  def getEvaluatedPopulation: Seq[EvaluatedResult] = _evaluatedEvolvedPopulation

  def getLastEvaluatedPopulation(workingDF: DataFrame, evaluationContextInfo: EvaluationContextInfo): Seq[EvaluatedResult] = {
    val newWorkingDFSize = workingDF.count()
    val evaluated = if (getEvaluatedPopulation.nonEmpty && newWorkingDFSize == currentWorkingDFSize) {
      debug(s"Taking $dimensionName evaluated population from previous generation.")
      getEvaluatedPopulation
    } else {
      if (newWorkingDFSize != currentWorkingDFSize && currentWorkingDFSize != 0) {
        debug(s"Reevaluating $dimensionName population due to increased working dataset size from $currentWorkingDFSize to $newWorkingDFSize")
        currentWorkingDFSize = newWorkingDFSize
        evaluatePopulation(getPopulation, workingDF, evaluationContextInfo)
      } else {
        currentWorkingDFSize = newWorkingDFSize
        debug(s"Evaluating $dimensionName population for the very first time.")
        evaluatePopulation(getPopulation, workingDF, evaluationContextInfo)
      }
    }
    evaluated
  }
}





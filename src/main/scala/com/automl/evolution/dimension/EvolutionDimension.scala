package com.automl.evolution.dimension

import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.template.{TemplateMember, TemplateTree}
import com.automl.{Evaluated, EvaluatedTemplateData, Population}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.param.Params
import org.apache.spark.sql.DataFrame

import scala.collection.mutable


/**
  * For fast implementions of coevolutions
  *
  * @tparam T
  */
//TODO problemType: ProblemType parameter might be moved somewhere to a field like `population`
trait EvolutionDimension[PopulationType <: Population[T], T, EvaluatedResult <: Evaluated] extends LazyLogging{

  var _population: PopulationType

  implicit val individualsEvaluationCache:mutable.Map[(T, Long), EvaluatedResult#FitnessType] //cache store anyone regardless of performance and could be/should be capped to prevent OOM

  val hallOfFame: mutable.PriorityQueue[EvaluatedResult]

  // Template pattern
  def evolve(population: PopulationType, workingDF: DataFrame): PopulationType = {
    val evaluatedOriginalPopulation = evaluatePopulation(population, workingDF)

    logger.debug("Selecting parents:")
    val selectedParents= selectParents(evaluatedOriginalPopulation)

    val selectedParentsPopulation = extractIndividualsFromEvaluatedIndividuals(selectedParents)

    logger.debug("Mutating parents:")
    val offspring = mutateParentPopulation(selectedParentsPopulation)

    logger.debug("Evaluating offspring:")
    val evaluatedOffspring = evaluatePopulation(offspring, workingDF)

    updateHallOfFame(evaluatedOffspring)

    val survivedForNextGenerationEvaluatedTemplates = selectSurvived(population.size, evaluatedOffspring)

    val evolvedNewGeneration = extractIndividualsFromEvaluatedIndividuals(survivedForNextGenerationEvaluatedTemplates)

    _population = evolvedNewGeneration
    evolvedNewGeneration
  }
  def updateHallOfFame(evaluatedIndividuals: Seq[EvaluatedResult]):Unit

  def getBestFromHallOfFame:EvaluatedResult = hallOfFame.head

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





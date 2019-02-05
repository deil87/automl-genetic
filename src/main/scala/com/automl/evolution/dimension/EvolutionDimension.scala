package com.automl.evolution.dimension

import com.automl.problemtype.ProblemType
import com.automl.{EvaluatedTemplateData, Population}
import org.apache.spark.ml.param.Params
import org.apache.spark.sql.DataFrame


/**
  * For implementing idea of coevolutions
  *
  * @tparam T
  */
//TODO problemType: ProblemType parameter might be moved somewhere to a field like `population`
trait EvolutionDimension[PopulationType <: Population[T], T, EvaluatedResult] {

  var _population: PopulationType

  def evolve(population: PopulationType, workingDF: DataFrame): PopulationType

  def selectParents(evaluatedIndividuals: Seq[EvaluatedResult]):  Seq[EvaluatedResult]

  def mutateParentPopulation(population: PopulationType): PopulationType

  def evaluatePopulation(population: PopulationType, workingDF: DataFrame): Seq[EvaluatedResult]

  def getPopulation: PopulationType = _population

  def getBestFromPopulation(workingDF: DataFrame): EvaluatedResult

}





package com.automl.evolution.dimension

import com.automl.{EvaluatedTemplateData, Population}
import org.apache.spark.ml.param.Params
import org.apache.spark.sql.DataFrame


/**
  * For implementing idea of coevolutions
  * @tparam T
  */
trait EvolutionDimension[PopulationType <: Population[T], T, EvaluatedResult] {

  var _population: PopulationType

  def evolve(population: PopulationType, workingDF: DataFrame): PopulationType

  def mutateParentPopulation(population: PopulationType): PopulationType

  def evaluatePopulation(population: PopulationType, workingDF: DataFrame): Seq[EvaluatedResult]

  def getPopulation: PopulationType = _population

  def getBestFromPopulation(workingDF: DataFrame): EvaluatedResult

}





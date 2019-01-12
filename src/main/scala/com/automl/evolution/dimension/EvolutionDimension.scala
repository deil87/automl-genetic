package com.automl.evolution.dimension

import com.automl.{EvaluatedTemplateData, Population}
import org.apache.spark.ml.param.Params
import org.apache.spark.sql.DataFrame

trait EvolutionDimension[T] {

  //TODO we need generic class for Population.
  def evolve(population: Population[T], workingDF: DataFrame): (Population[T], Option[EvaluatedTemplateData])

  // We need to decompose evaluation as well.  That way we will be able not to return tuple.
  def evaluateDimension()

  def applyMutation()

  def getBestPopulation(): Map[String, Seq[Params]] = Map.empty
}





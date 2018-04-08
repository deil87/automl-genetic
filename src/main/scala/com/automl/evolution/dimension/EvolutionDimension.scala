package com.automl.evolution.dimension

import com.automl.{EvaluatedTemplateData, Population}
import org.apache.spark.ml.param.Params
import org.apache.spark.sql.DataFrame

trait EvolutionDimension {

  def evolve(population: Population, workingDF: DataFrame): (Population, Option[EvaluatedTemplateData])
  def applyMutation()

  def getBestPopulation(): Map[String, Seq[Params]] = Map.empty
}





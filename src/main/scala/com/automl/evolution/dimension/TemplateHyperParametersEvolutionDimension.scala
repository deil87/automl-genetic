package com.automl.evolution.dimension
import com.automl.Population
import org.apache.spark.ml.param.Params
import org.apache.spark.sql.DataFrame

class TemplateHyperParametersEvolutionDimension extends EvolutionDimension {

  override def evolve(population: Population, workingDF: DataFrame) = ???

  override def applyMutation(): Unit = ???

  override def getBestPopulation(): Map[String, Seq[Params]] = Map.empty
}
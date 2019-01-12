package com.automl.evolution.dimension
import com.automl.TPopulation
import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.ml.param.Params
import org.apache.spark.sql.DataFrame

// it should be a HYPER DIMENSION. We want to find the best FIELD of hyper parameters here.
class TemplateHyperParametersEvolutionDimension(evolveEveryGenerations: Int = 1) extends EvolutionDimension[HyperParametersField] {

  var hyperParametersMap: Map[String, Seq[Params]]  = Map.empty

  override def evolve(population: TPopulation, workingDF: DataFrame) = {
    val defaultConfig = ConfigFactory.load()
    val numberOfHPEvolutionsPerGeneration = defaultConfig.getInt("evolution.hyperParameterDimension.numOfEvolutionsPerGen")

  }

  override def applyMutation(): Unit = ???

  override def getBestPopulation(): Map[String, Seq[Params]] = hyperParametersMap
}

//We need to evolve population of parameters for every model individually. So we will span multiple coevolutions per Model.

class HyperParametersField()
class ModelHyperParameters(val modelKey: String,  val hpGroup: HyperParametersGroup)
class HyperParametersGroup( params: Seq[Params])
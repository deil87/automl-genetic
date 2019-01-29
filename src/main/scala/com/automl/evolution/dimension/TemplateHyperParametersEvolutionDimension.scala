package com.automl.evolution.dimension
import com.automl.Population
import com.automl.problemtype.ProblemType
import com.typesafe.config.ConfigFactory
import org.apache.spark.ml.param.{ParamPair, Params}
import org.apache.spark.sql.DataFrame

// it should be a HYPER DIMENSION. We want to find the best FIELD of hyper parameters here.
//It is right thing that I have removed problemType from methods parameters of EvolutionDimension as not all dimensions are going to be dependant on this.
//If we need dimension to depend we can pass parameter to a constructor of the dimension's class
class TemplateHyperParametersEvolutionDimension(evolveEveryGenerations: Int = 1) extends EvolutionDimension[HPPopulation, HyperParametersField, EvaluatedHyperParametersField] {

  override var _population: HPPopulation = _

  override def evolve(population: HPPopulation, workingDF: DataFrame): HPPopulation = {
    val defaultConfig = ConfigFactory.load()
    val numberOfHPEvolutionsPerGeneration = defaultConfig.getInt("evolution.hyperParameterDimension.numOfEvolutionsPerGen")
    val initialPopulation = Seq(
      new HyperParametersField(
        Seq(
          new ModelHyperParameters("DRF", new HyperParametersGroup(???))
        )
      )
    )
    new HPPopulation(individuals = Nil)
  }

  override def mutateParentPopulation(population: HPPopulation): HPPopulation = ???

  override def evaluatePopulation(population: HPPopulation, workingDF: DataFrame): Seq[EvaluatedHyperParametersField] = ???

  override def getBestFromPopulation(workingDF: DataFrame): EvaluatedHyperParametersField = ???
}

//We need to evolve population of parameters for every model individually. So we will span multiple coevolutions per Model.

class HyperParametersField(modelParameters: Seq[ModelHyperParameters])
class ModelHyperParameters(val modelKey: String,  val hpGroup: HyperParametersGroup)
class HyperParametersGroup( params: Params)

class HPPopulation(val individuals: Seq[ HyperParametersField]) extends Population[HyperParametersField]

case class EvaluatedHyperParametersField(field: HyperParametersField)
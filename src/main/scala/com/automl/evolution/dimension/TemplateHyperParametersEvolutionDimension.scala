package com.automl.evolution.dimension
import com.automl.Population
import com.automl.problemtype.ProblemType
import com.automl.template.simple.Bayesian
import com.typesafe.config.ConfigFactory
import org.apache.spark.ml.param.{ParamPair, Params}
import org.apache.spark.sql.DataFrame

import scala.util.Random

// it should be a HYPER DIMENSION. We want to find the best FIELD of hyper parameters here.
//It is right thing that I have removed problemType from methods parameters of EvolutionDimension as not all dimensions are going to be dependant on this.
//If we need dimension to depend we can pass parameter to a constructor of the dimension's class
class TemplateHyperParametersEvolutionDimension(evolveEveryGenerations: Int = 1) extends EvolutionDimension[HPPopulation, HyperParametersField, EvaluatedHyperParametersField] {

  override var _population: HPPopulation = _

  val defaultConfig = ConfigFactory.load()
  val numberOfHPEvolutionsPerGeneration = defaultConfig.getInt("evolution.hyperParameterDimension.numOfEvolutionsPerGen")

  //TODO consider moving this to EvolutionDimension interface
  def evolveInitial(workingDF: DataFrame):HPPopulation = {
    val initialPopulation = Seq(
      HyperParametersField(
        Seq(
          //          new ModelHyperParameters("LogisticRegression", new HyperParametersGroup(???)),
          BayesianHPGroup()
        )
      ), HyperParametersField(
        Seq(
          BayesianHPGroup()
        )
      )
    )
    evolve(new HPPopulation(individuals = initialPopulation), workingDF)
  }

  // TODO we need last population of templates or pool of templates per model(we are evolving parameters for)

//  override def evolve(population: HPPopulation, workingDF: DataFrame): HPPopulation = {
//
//    val evaluatedOriginalPopulation = evaluatePopulation(population, workingDF)
//
//    val selectedParents = selectParents(evaluatedOriginalPopulation)
//
//    val selectedParentsPopulation = extractIndividualsFromEvaluated(selectedParents)
//
//    val offspring = mutateParentPopulation(selectedParentsPopulation)
//
//    val evaluationsForOffspring = evaluatePopulation(offspring, workingDF)
//
//    val survivedForNextGenerationEvaluatedTemplates = selectSurvived(population.size, evaluationsForOffspring)
//
//    extractIndividualsFromEvaluated(survivedForNextGenerationEvaluatedTemplates)
//  }



  override def extractIndividualsFromEvaluatedIndividuals(evaluatedIndividuals: Seq[EvaluatedHyperParametersField]): HPPopulation = {
    new HPPopulation(evaluatedIndividuals.map(_.field))
  }

  override def selectParents(evaluatedIndividuals: Seq[EvaluatedHyperParametersField]): Seq[EvaluatedHyperParametersField] = {
    val numberOfParents = 3 // read it from config
    evaluatedIndividuals.sortWith(_.score > _.score).take(numberOfParents)

  }

  override def selectSurvived(populationSize: Int, evaluatedIndividuals: Seq[EvaluatedHyperParametersField]): Seq[EvaluatedHyperParametersField] = {
    // TODO reimplement with RankSelectionStrategy
    evaluatedIndividuals.sortWith(_.score > _.score).take(populationSize)
  }


  override def mutateParentPopulation(population: HPPopulation): HPPopulation = {
    // Use lenses here :)
    new HPPopulation(
      population.individuals.map { hpField => {
        hpField.copy(modelsHParameterGroups = hpField.modelsHParameterGroups.map { hpGroup => {
          val mutatedGroup = hpGroup  // maybe better stick to .copy() approach all the way down
          mutatedGroup.mutate()
          hpGroup
        }
        })
      }
    })
  }

  override def evaluatePopulation(population: HPPopulation, workingDF: DataFrame): Seq[EvaluatedHyperParametersField] = {
    val Array(trainingSplit, testSplit) = workingDF.randomSplit(Array(0.67, 0.33), 11L) // TODO move to Config ratio
    population.individuals.map { hpField =>
      val metricsFromAllModelsEvaluations = hpField.modelsHParameterGroups.map {
        case hpGroup@BayesianHPGroup() =>
          val metric = Bayesian(hpGroup).fitnessError(trainingSplit, testSplit).getCorrespondingMetric
          metric
        case _ => ???
      }
      EvaluatedHyperParametersField(hpField, metricsFromAllModelsEvaluations.sum)
    }
  }

  override def getBestFromPopulation(workingDF: DataFrame): EvaluatedHyperParametersField = ???
}

//We need to evolve population of parameters for every model individually. So we will span multiple coevolutions (each one per Model).

case class HyperParametersField(modelsHParameterGroups: Seq[HyperParametersGroup[_]])

trait HyperParametersGroup[HPModelBoundedType <: MutableHParameter[Double, HPModelBoundedType]]{
  var hpParameters: Seq[HPModelBoundedType]
  def mutate(): HyperParametersGroup[HPModelBoundedType]
}//TODO instead of using Any we can create our own hierarhy of wrapper classes to make them have coomon ancestor like ParameterType

case class BayesianHPGroup() extends HyperParametersGroup[BayesianHParameter[Double]] { // We need F-bounded polymorphism here to make sure we use only corresponding parameters inside group
  var hpParameters:Seq[BayesianHParameter[Double]] = Seq(Smoothing())// TODO consider using HList
  override def mutate(): HyperParametersGroup[BayesianHParameter[Double]] = {
    val mutated = hpParameters.map(hpModelTpe => hpModelTpe.mutate())
    hpParameters = mutated
    this
  }
}
object BayesianHPGroup {
  val default = BayesianHPGroup()
}


trait HPRange[RangeType <: AnyVal] {
  def min: RangeType
  def max: RangeType
  def step: RangeType
}
trait DoubleHPRange extends HPRange[Double]

sealed trait HParameter[+T] {
  def getDefault:T
}

trait MutableHParameter[+T, V <: MutableHParameter[T, V]] extends HParameter[T] {
  def currentValue: T
  def mutate(): V
}

//trait ModelSpecificHP[+T] extends MutableHParameter[T]

trait BayesianHParameter[T <: AnyVal] extends MutableHParameter[T, BayesianHParameter[T]]

case class Smoothing() extends BayesianHParameter[Double] with DoubleHPRange { // we can specialize with Marker trait which parameter can be used with which Model
  override def min: Double = 1.0

  override def max: Double = 3.0

  override def step: Double = 1.0

  override def getDefault: Double = 1.0

  var currentValue: Double = getDefault

  override def mutate(): Smoothing = {
    var newValue = getNextWithinTheRange
    while(newValue == currentValue) {
      newValue = getNextWithinTheRange
    }
    val newVersion = Smoothing()
    newVersion.currentValue = newValue
    newVersion
  }

  private def getNextWithinTheRange = {
    new Random().nextInt(max.toInt) + min
  }
}



class HPPopulation(val individuals: Seq[ HyperParametersField]) extends Population[HyperParametersField]

case class EvaluatedHyperParametersField(field: HyperParametersField, score:Double)
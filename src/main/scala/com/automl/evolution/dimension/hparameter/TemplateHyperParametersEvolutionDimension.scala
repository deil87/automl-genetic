package com.automl.evolution.dimension.hparameter


import com.automl.Population
import com.automl.problemtype.ProblemType
import com.automl.template.simple.{Bayesian, LogisticRegressionModel}
import com.typesafe.config.ConfigFactory
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.sql.DataFrame
import com.automl.Evaluated
import com.automl.evolution.dimension.{EvolutionDimension, TemplateEvolutionDimension}
import utils.BenchmarkHelper

import scala.collection.mutable
import scala.util.Random

// it should be a HYPER DIMENSION. We want to find the best FIELD of hyper parameters here.
//It is right thing that I have removed problemType from methods parameters of EvolutionDimension as not all dimensions are going to be dependant on this.
//If we need dimension to depend we can pass parameter to a constructor of the dimension's class
class TemplateHyperParametersEvolutionDimension(parentTemplateEvDimension: TemplateEvolutionDimension, evolveEveryGenerations: Int = 1, problemType: ProblemType)
  extends EvolutionDimension[HPPopulation, HyperParametersField, EvaluatedHyperParametersField] with LazyLogging{

  override var _population: HPPopulation = _

  val hpdConfig = ConfigFactory.load().getConfig("evolution.hyperParameterDimension")

  lazy val evolutionDimensionLabel: String = hpdConfig.getString("name")

  lazy val numberOfHPEvolutionsPerGeneration: Int = hpdConfig.getInt("numOfEvolutionsPerGen")

  override implicit val individualsEvaluationCache: mutable.Map[(HyperParametersField, Long), Double] = mutable.Map[(HyperParametersField, Long), Double]()


  override val hallOfFame: mutable.PriorityQueue[EvaluatedHyperParametersField] = collection.mutable.PriorityQueue[EvaluatedHyperParametersField]()


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

// Use implicits to mutate HyperParametersField
  override def mutateParentPopulation(population: HPPopulation): HPPopulation = {
    // Use lenses here :)
    new HPPopulation(
      population.individuals.map { hpField => {
        val newField = HyperParametersField(modelsHParameterGroups = hpField.modelsHParameterGroups.map { hpGroup => hpGroup.mutate() })
        require(hpField.hashCode() != newField.hashCode(), "Hash codes should be different")
        logger.debug(s"HyperParametersField mutated from ${hpField.modelsHParameterGroups.map(_.hpParameters.map(_.currentValue).mkString(","))} to ${newField.modelsHParameterGroups.map(_.hpParameters.map(_.currentValue).mkString(","))}")
        newField
      }
    })
  }

  // TODO we need last population of templates or pool of templates per model(we are evolving parameters for)
  override def evaluatePopulation(population: HPPopulation, workingDF: DataFrame): Seq[EvaluatedHyperParametersField] = {

    val numberOfBestTemplates = 3
    // Note: there are multiple strategies of evaluating hps for template population.
    // 1) estimate base model/ building blocks of the templates(ensembles)
    // 2) estimate on last survived population(part of it)
    // 3) estimate on hallOfFame models (but we more conserned about population we are evolving at hands)
    // 4) ideally we need tu estimate parameters for all possible positions of the models in the ensembles.... this is for the future versions.
    // 5) mixture of 3) and 1)
    BenchmarkHelper.time("Hyper-parameter evaluatePopulation ") {
      val threeBestTemplates = parentTemplateEvDimension.getEvaluatedPopulation.sortWith((a, b) => a.fitness.orderTo(b.fitness)).map(_.template).take(numberOfBestTemplates)

      val Array(trainingSplit, testSplit) = workingDF.randomSplit(Array(0.67, 0.33), 11L) // TODO move to Config ratio
      population.individuals.map { hpField =>
        val cacheKey = (hpField, workingDF.count())
        val cacheKeyHashCode = cacheKey.hashCode()
        if (individualsEvaluationCache.isDefinedAt(cacheKey)) {
          logger.debug(s"Cache hit happened for individual: $hpField")
          logger.debug(s"Retrieved value from the cache with hashCode = $cacheKeyHashCode : ${individualsEvaluationCache(cacheKey)}")
        }
        val fitness = individualsEvaluationCache.getOrElseUpdate(cacheKey, {
          logger.debug(s"Entry with hashCode = ${cacheKey.hashCode()} was added to the cache.")
          // Estimating 1) building blocks
          logger.debug(s"Evaluating hpfield on base models:")
          val metricsFromBaseModels = hpField.modelsHParameterGroups.map {
            case hpGroup@BayesianHPGroup(_) =>
              val metric = Bayesian(hpGroup).fitnessError(trainingSplit, testSplit, problemType).getCorrespondingMetric
              // We should get last Best Population from the TemplateCoevolution and estimate on the whole population or representative sample
              metric
            case hpGroup@LogisticRegressionHPGroup(_) =>
              val metric = LogisticRegressionModel(hpGroup).fitnessError(trainingSplit, testSplit, problemType).getCorrespondingMetric
              metric
            case _ => ???
          }
          // Estimating 2)
          logger.debug(s"Evaluating hpfield on ${threeBestTemplates.size} best templates in current population:")
          val threeBestEvaluations = threeBestTemplates.map(template => template.evaluateFitness(trainingSplit, testSplit, problemType, hyperParamsMap = hpField).getCorrespondingMetric)
          metricsFromBaseModels.sum + threeBestEvaluations.sum // we sum all metrics from each ModelHPGroup inn the field so that we can later decide which Field is the best
        })
        EvaluatedHyperParametersField(hpField, fitness)
      }
    }
  }

  def updateHallOfFame(evaluatedIndividuals: Seq[EvaluatedHyperParametersField]):Unit = {
    val hallOfFameUpdateSize = 5  // TODO Config
    hallOfFame.headOption.map{bestAtAllTimes =>
      evaluatedIndividuals.filter(_.score >= bestAtAllTimes.score).take(hallOfFameUpdateSize).foreach(ev => hallOfFame.enqueue(ev)) //TODO Too many duplicate entries in the queue!!!
    }.getOrElse{
      evaluatedIndividuals.take(hallOfFameUpdateSize).foreach(ev => hallOfFame.enqueue(ev))
//      hallOfFame ++ evaluatedIndividuals.take(hallOfFameUpdateSize) //TODO why it does not put elements?
    }
  }


  override def getInitialColdStartPopulation: HPPopulation = {
    new HPPopulation(Seq(
      HyperParametersField(
        Seq(
          BayesianHPGroup()
        )
      ), HyperParametersField(
        Seq(
          BayesianHPGroup()
        )
      )
    ))
  }

  //Note: if we don't have hallOfFame filled up then we want to just take any HyperParametersField field to start with ( we don't have hyper parameters evaluated yet)
  // WE can start Template evolution with default hyper parameters because evaluation of hyper parameters on first iteration contributes only
  // to the upcoming evolutions as for the first iteration we have nothing to compare against for HyperParameter dimension //TODO make it configurable strategy
  //But it makes sense to search for HPs first. It is faster and brings more value
  override def getBestFromHallOfFame: HyperParametersField = hallOfFame.headOption.map(_.field).getOrElse{getInitialPopulation.individuals.head}

  override def getBestFromPopulation(workingDF: DataFrame): EvaluatedHyperParametersField = {
    logger.debug("Getting best individual from population...")
    evaluatePopulation(getPopulation, workingDF).sortWith(_.score > _.score).head // TODO check that it might be stored already in a sorted way
  }
}

//We need to evolve population of parameters for every model individually. So we will span multiple coevolutions (each one per Model).

//case class HyperParametersField[T  <: MutableHParameter[Double, T]](modelsHParameterGroups: Seq[HyperParametersGroup[ T]]) {
//case class HyperParametersField(modelsHParameterGroups: Seq[HyperParametersGroup[MutableHParameter[Double, _]]]) {
//case class HyperParametersField(modelsHParameterGroups: Seq[HyperParametersGroup[ExT] forSome { type Ext <: MutableHParameter[Double, _]}] ) {
// Existential Types - I love you guys!
case class HyperParametersField(modelsHParameterGroups: Seq[HyperParametersGroup[_ <: MutableHParameter[Double, _]]] ) {

  override def hashCode(): Int = {
    modelsHParameterGroups.map(_.hpParameters.map(_.currentValue.hashCode()).sum).sum
  }

  def getLogisticRegressionHPGroup: HyperParametersGroup[_ <: MutableHParameter[Double, _]] = {
    modelsHParameterGroups.find{
      case LogisticRegressionHPGroup(_) => true
    }.get
  }
}

object HyperParametersField {
  def default = HyperParametersField(Seq(BayesianHPGroup(), LogisticRegressionHPGroup()))
}

//Подумать над тем чтобы использовать обычный Map
trait HyperParametersGroup[HPModelBoundedType <: MutableHParameter[Double, HPModelBoundedType]]{
  def hpParameters : Seq[HPModelBoundedType]
  def mutate(): HyperParametersGroup[HPModelBoundedType]
} //TODO instead of using Any we can create our own hierarhy of wrapper classes to make them have coomon ancestor like ParameterType


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


class HPPopulation(val individuals: Seq[ HyperParametersField]) extends Population[HyperParametersField]

case class EvaluatedHyperParametersField(field: HyperParametersField, score:Double) extends Evaluated {
  override type FitnessType = Double
}

object EvaluatedHyperParametersField {
  implicit val individualsOrdering: Ordering[EvaluatedHyperParametersField] = new Ordering[EvaluatedHyperParametersField] {
    override def compare(x: EvaluatedHyperParametersField, y: EvaluatedHyperParametersField): Int = {
      x.score.compareTo(y.score) // TODO swap y and x depending on what is needed in the Priority queue
    }
  }
}
package com.automl.evolution.dimension
import java.util.UUID

import com.automl.Population
import com.automl.problemtype.ProblemType
import com.automl.template.simple.Bayesian
import com.typesafe.config.ConfigFactory
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.param.{ParamPair, Params}
import org.apache.spark.sql.DataFrame
import com.automl.Evaluated
import com.automl.helper.FitnessResult
import com.automl.template.{TemplateMember, TemplateTree}

import scala.collection.mutable
import scala.util.Random

// it should be a HYPER DIMENSION. We want to find the best FIELD of hyper parameters here.
//It is right thing that I have removed problemType from methods parameters of EvolutionDimension as not all dimensions are going to be dependant on this.
//If we need dimension to depend we can pass parameter to a constructor of the dimension's class
class TemplateHyperParametersEvolutionDimension(evolveEveryGenerations: Int = 1, problemType: ProblemType)
  extends EvolutionDimension[HPPopulation, HyperParametersField, EvaluatedHyperParametersField] with LazyLogging{

  override var _population: HPPopulation = _

  val defaultConfig = ConfigFactory.load()
  val numberOfHPEvolutionsPerGeneration = defaultConfig.getInt("evolution.hyperParameterDimension.numOfEvolutionsPerGen")

  override implicit val individualsEvaluationCache: mutable.Map[(HyperParametersField, Long), Double] = mutable.Map[(HyperParametersField, Long), Double]()


  override val hallOfFame: mutable.PriorityQueue[EvaluatedHyperParametersField] = collection.mutable.PriorityQueue[EvaluatedHyperParametersField]()

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

// Use implicits to mutate HyperParametersField
  override def mutateParentPopulation(population: HPPopulation): HPPopulation = {
    // Use lenses here :)
    new HPPopulation(
      population.individuals.map { hpField => {
        logger.debug(s"Hash code before ${hpField.hashCode()} for value ${hpField.modelsHParameterGroups.map(_.hpParameters.map(_.currentValue).mkString(","))}")
        val newField = HyperParametersField(modelsHParameterGroups = hpField.modelsHParameterGroups.map { hpGroup => hpGroup.mutate() })
        logger.debug(s"Hash code after ${newField.hashCode()} for value ${newField.modelsHParameterGroups.map(_.hpParameters.map(_.currentValue).mkString(","))}")
        newField

        /*hpField.copy(modelsHParameterGroups = hpField.modelsHParameterGroups.map {
          case hpGroup@BayesianHPGroup(_) => BayesianHPGroup(hpGroup.mutate().hpParameters )})*/
      }
    })
  }

  override def evaluatePopulation(population: HPPopulation, workingDF: DataFrame): Seq[EvaluatedHyperParametersField] = {

    val Array(trainingSplit, testSplit) = workingDF.randomSplit(Array(0.67, 0.33), 11L) // TODO move to Config ratio
    population.individuals.map { hpField =>
      val cacheKey = (hpField, workingDF.count())
      val cacheKeyHashCode = cacheKey.hashCode()
      if (individualsEvaluationCache.isDefinedAt(cacheKey)) {
        logger.debug(s"Hyper parameter evolution. Cache hit happened for individual: $hpField")
        logger.debug(s"Retrieved value from the cache with hashCode = $cacheKeyHashCode : ${individualsEvaluationCache(cacheKey)}")
      }
      val fitness = individualsEvaluationCache.getOrElseUpdate(cacheKey, {
        logger.debug(s"Entry with hashCode = ${cacheKey.hashCode()} was added to the cache.")
        val metricsFromAllModelsEvaluations = hpField.modelsHParameterGroups.map {
          case hpGroup@BayesianHPGroup(_) =>
            val metric = Bayesian(hpGroup).fitnessError(trainingSplit, testSplit, problemType).getCorrespondingMetric
            metric
          case _ => ???
        }
         metricsFromAllModelsEvaluations.sum
      })
      EvaluatedHyperParametersField(hpField, fitness)
    }
  }

  def updateHallOfFame(evaluatedIndividuals: Seq[EvaluatedHyperParametersField]):Unit = {
    val hallOfFameUpdateSize = 5  // TODO Config
    hallOfFame.headOption.map{bestAtAllTimes =>
      evaluatedIndividuals.filter(_.score >= bestAtAllTimes.score).take(hallOfFameUpdateSize).foreach(ev => hallOfFame.enqueue(ev))
    }.getOrElse{
      evaluatedIndividuals.take(hallOfFameUpdateSize).foreach(ev => hallOfFame.enqueue(ev))
//      hallOfFame ++ evaluatedIndividuals.take(hallOfFameUpdateSize) //TODO why it does not put elements?
    }

  }

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
}

//Подумать над тем чтобы использовать обычный Map
trait HyperParametersGroup[HPModelBoundedType <: MutableHParameter[Double, HPModelBoundedType]]{
  def hpParameters : Seq[HPModelBoundedType]
  def mutate(): HyperParametersGroup[HPModelBoundedType]
} //TODO instead of using Any we can create our own hierarhy of wrapper classes to make them have coomon ancestor like ParameterType

case class BayesianHPGroup(hpParameters:Seq[BayesianHParameter[Double]] = Seq(Smoothing()))
                extends HyperParametersGroup[BayesianHParameter[Double]] {

  // TODO consider using HList
  override def mutate(): HyperParametersGroup[BayesianHParameter[Double]] = {
    BayesianHPGroup(hpParameters = hpParameters.map(hpModelTpe => hpModelTpe.mutate()))
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

  override def max: Double = 10.0

  override def step: Double = 1.0

  override def getDefault: Double = 2.0

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

  override def toString: String = "smoothing:" + currentValue.toString
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
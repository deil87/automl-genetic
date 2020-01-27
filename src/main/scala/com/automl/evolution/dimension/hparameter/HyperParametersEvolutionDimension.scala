package com.automl.evolution.dimension.hparameter


import com.automl.{ConfigProvider, Evaluated, PaddedLogging}
import com.automl.problemtype.ProblemType
import org.apache.spark.sql.DataFrame
import com.automl.evolution.dimension.{EvolutionDimension, TemplateEvolutionDimension}
import com.automl.evolution.evaluation.{EvaluationContextInfo, HyperParameterContemporaryPopulationEvaluator, HyperParameterMixedEvaluator}
import com.automl.evolution.mutation.{DepthDependentTemplateMutationStrategy, HPMutationStrategy}
import com.automl.helper.PopulationHelper
import com.automl.population.{GenericPopulationBuilder, HPPopulation}
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.{TemplateMember, TemplateTree}
import com.typesafe.scalalogging.LazyLogging

import scala.collection.mutable
import scala.math.BigDecimal.RoundingMode
import scala.util.Random

// it should be a HYPER DIMENSION. We want to find the best FIELD of hyper parameters here.
//It is right thing that I have removed problemType from methods parameters of EvolutionDimension as not all dimensions are going to be dependant on this.
//If we need dimension to depend we can pass parameter to a constructor of the dimension's class
class HyperParametersEvolutionDimension(parentTemplateEvDimension: TemplateEvolutionDimension, evolveEveryGenerations: Int = 1, problemType: ProblemType, seed: Long)(implicit val logPaddingSize: Int)
  extends EvolutionDimension[HPPopulation, HyperParametersField, EvaluatedHyperParametersField] with PaddedLogging{

  override def dimensionName: String = "HyperparametersDimension"

  override var _population: HPPopulation = new HPPopulation(Nil)

  val hpdConfig = ConfigProvider.config.getConfig("evolution.hyperParameterDimension")

  lazy val evolutionDimensionLabel: String = hpdConfig.getString("name")

  val hpMutationStrategy = new HPMutationStrategy()(logPaddingSize + 4)

  lazy val numberOfHPEvolutionsPerGeneration: Int = hpdConfig.getInt("numOfEvolutionsPerGen")
  lazy val populationSize: Int = hpdConfig.getInt("populationSize")

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
  override def mutateParentPopulation(population: HPPopulation, notToIntersectWith: HPPopulation): HPPopulation = {
    hpMutationStrategy.mutate(population)
  }


  override def showCurrentPopulation(): Unit = { // TODO generalize
      if(getEvaluatedPopulation.nonEmpty)
        debug(PopulationHelper.renderEvaluatedIndividuals(getEvaluatedPopulation))
      else {}
  }

  override def evaluatePopulation(population: HPPopulation, workingDF: DataFrame, evaluationContextInfo: EvaluationContextInfo): Seq[EvaluatedHyperParametersField] = {

    new HyperParameterContemporaryPopulationEvaluator(parentTemplateEvDimension)(logPaddingSize + 4)
      .evaluateIndividuals(population, workingDF, problemType, evaluationContextInfo, seed)
//    new HyperParameterMixedEvaluator(parentTemplateEvDimension)(logPaddingSize + 4).evaluateIndividuals(population, workingDF, problemType)
  }

  override def getInitialPopulationFromMetaDB: HPPopulation = ???

  override def getInitialColdStartPopulation: HPPopulation = {
    //    initialPopulation.map{population => TODO for now we have hardcoded initialPopulationm but might want to pass a parameter from TemplateDimension

    val initialPopulation = new HPPopulation(Seq(
      HyperParametersField(
        Seq(
          BayesianHPGroup(), LogisticRegressionHPGroup(), DecisionTreeHPGroup()
        )
      ), HyperParametersField(
        Seq(
          BayesianHPGroup(), LogisticRegressionHPGroup(), DecisionTreeHPGroup()
        )
      )
    ))
    val population = GenericPopulationBuilder.fromSeedPopulation(initialPopulation)
      .withSize(populationSize)
      .build
    population
  }

  //Note: if we don't have hallOfFame filled up then we want to just take any HyperParametersField field to start with ( we don't have hyper parameters evaluated yet)
  // WE can start Template evolution with default hyper parameters because evaluation of hyper parameters on first iteration contributes only
  // to the upcoming evolutions as for the first iteration we have nothing to compare against for HyperParameter dimension //TODO make it configurable strategy
  //But it makes sense to search for HPs first. It is faster and brings more value
  override def getBestFromHallOfFame: HyperParametersField =
    hallOfFame.headOption.map(_.field)
//      .getOrElse{getInitialPopulation.individuals.randElement}
      .getOrElse{throw new IllegalStateException("HP coevolution did not produce any HPField yet. For researc purposes it is better not to return randElement")}
}


// TODO Подумать над тем чтобы использовать обычный Map
trait HyperParametersGroup[HPModelBoundedType <: MutableHParameter[Double, HPModelBoundedType]]{
  def hpParameters : Seq[HPModelBoundedType]
  def mutate(): HyperParametersGroup[HPModelBoundedType]
  //TODO maybe it is better to use Map as for now we have only one HPGroup per model
  def isRelevantTo(template: TemplateMember): Boolean

  override def equals(obj: Any): Boolean = {
    require(obj.isInstanceOf[HyperParametersGroup[HPModelBoundedType]])
    val another = obj.asInstanceOf[HyperParametersGroup[HPModelBoundedType]]
    val diff = this.hpParameters.map(hp => (hp.currentValue, hp.getClass)).diff(another.hpParameters.map(hp => (hp.currentValue, hp.getClass)))
    diff.isEmpty
  }
} //TODO instead of using Any we can create our own hierarhy of wrapper classes to make them have coomon ancestor like ParameterType


trait HPRange[RangeType <: AnyVal] {
  def min: RangeType
  def max: RangeType
  def step: RangeType
  def numberOfEntries: Int
}

trait DoubleHPRange[V <: MutableHParameter[Double, V]] extends HPRange[Double] with LazyLogging{ this: MutableHParameter[Double, V] =>

  val config = ConfigProvider.config.getConfig("evolution")
  val randomMutationFrequency = config.getDouble("hyperParameterDimension.randomMutationFrequency")

  val numberOfEntries: Int = {
    val d = (max - min + step) / step
    Math.round(d).asInstanceOf[Int]
  }

  def round(value: Double, places: Int): Double = {
    if (places < 0) throw new IllegalArgumentException
    val bd = BigDecimal(value).setScale(places, RoundingMode.HALF_UP)
    bd.doubleValue
  }

  def getNextWithinTheRange: Double = {
    val range = BigDecimal(min) to BigDecimal(max) by BigDecimal(step)
    range(new Random().nextInt(range.length)).toDouble
  }

  def getNextClosestWithinTheRange(currentValue: Double): Double = {

    val mutated =
      if (new Random().nextDouble() < randomMutationFrequency) {
        logger.debug(s"Random jump will be performed for ${this.getClass} hps")
        getNextWithinTheRange
      }
      else if (new Random().nextDouble() < 0.5) {
        currentValue - step
      } else {
        currentValue + step
      }

    val next = if (mutated > max) min
    else if (mutated < min) max
    else mutated

    if(step < 1) round(next, 1) else next
    // TODO Add random jump based on frequency defined in the config
    //if(mutated <= max && mutated >= min) mutated else getNextClosestWithinTheRange(currentValue)
  }

  def newInstance: V

  var currentValue: Double = {
    val defaultValue = initialValue.getOrElse(getDefaultRandomly)
    defaultValue
  }

  override def mutate(): V = { // TODO could be simplified. Return Double type.
    if(min == max) {
      this.asInstanceOf[V]
    } else {
      var mutatedVersion = newInstance
      while (currentValue == mutatedVersion.currentValue) {
        mutatedVersion = newInstance
      }
      mutatedVersion
    }
  }

  def validate(value: Double): Unit = require(value <= max && value >=min, "Mutated value is outside of the allowed range")
}

sealed trait HParameter[+T] {
  def getDefaultRandomly:T
  def initialValue:Option[T]
}

trait MutableHParameter[T, V <: MutableHParameter[T, V]] extends HParameter[T] {
  var currentValue: T
  def mutate(): V

  override def toString: String = currentValue.toString
}

case class EvaluatedHyperParametersField(field: HyperParametersField, score:Double, problemType: ProblemType) extends Evaluated[EvaluatedHyperParametersField] {
  override type ItemType = HyperParametersField
  override type FitnessType = Double

  override type ParamsType = AnyVal //Unused

  override def idShort: String = s"idShort: ${field.hashCode}"

  override def params: Option[ParamsType] = None

  override def item: HyperParametersField = field
  override def result: Double = score


  override def betterThan(that: EvaluatedHyperParametersField): Boolean = ???

  override def betterThanOrEqual(that: EvaluatedHyperParametersField): Int = ???

  override def compare(other: EvaluatedHyperParametersField): Int = {
    val comparisonResult = score.compare(other.score)
    if(theBiggerTheBetter(problemType)) comparisonResult else (-1) * comparisonResult
  }
}

object EvaluatedHyperParametersField {
  //TODO unused
  implicit val individualsOrdering: Ordering[EvaluatedHyperParametersField] = new Ordering[EvaluatedHyperParametersField] {
    override def compare(x: EvaluatedHyperParametersField, y: EvaluatedHyperParametersField): Int = {
      x.score.compareTo(y.score) // TODO swap y and x depending on what is needed in the Priority queue - use reverse method
    }
  }
}
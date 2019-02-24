package com.automl.evolution.dimension.hparameter


import com.automl.{Evaluated, PaddedLogging, Population}
import com.automl.problemtype.ProblemType
import com.automl.template.simple.{Bayesian, DecisionTree, LogisticRegressionModel}
import com.typesafe.config.ConfigFactory
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.sql.DataFrame
import com.automl.dataset.StratifiedSampling
import com.automl.evolution.dimension.{EvolutionDimension, TemplateEvolutionDimension}
import com.automl.evolution.evaluation.HyperParameterPopulationEvaluator
import com.automl.helper.PopulationHelper
import utils.BenchmarkHelper

import scala.collection.mutable
import scala.math.BigDecimal.RoundingMode
import scala.util.Random

// it should be a HYPER DIMENSION. We want to find the best FIELD of hyper parameters here.
//It is right thing that I have removed problemType from methods parameters of EvolutionDimension as not all dimensions are going to be dependant on this.
//If we need dimension to depend we can pass parameter to a constructor of the dimension's class
class TemplateHyperParametersEvolutionDimension(parentTemplateEvDimension: TemplateEvolutionDimension, evolveEveryGenerations: Int = 1, problemType: ProblemType)(implicit val logPaddingSize: Int)
  extends EvolutionDimension[HPPopulation, HyperParametersField, EvaluatedHyperParametersField] with PaddedLogging{

  override var _population: HPPopulation = new HPPopulation(Nil)

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
        debug(s"HyperParametersField mutated from $hpField to $newField")
        newField
      }
    })
  }


  override def showCurrentPopulation(): Unit = { // TODO generalize
      if(getEvaluatedPopulation.nonEmpty)
        debug(PopulationHelper.renderEvaluatedIndividuals(getEvaluatedPopulation))
      else {}
  }

  override def evaluatePopulation(population: HPPopulation, workingDF: DataFrame): Seq[EvaluatedHyperParametersField] = {

    new HyperParameterPopulationEvaluator(parentTemplateEvDimension)(logPaddingSize + 4).evaluateIndividuals(population, workingDF, problemType)
  }

  override def getInitialPopulationFromMetaDB: HPPopulation = ???

  override def getInitialColdStartPopulation: HPPopulation = {
    new HPPopulation(Seq(
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
  }

  //Note: if we don't have hallOfFame filled up then we want to just take any HyperParametersField field to start with ( we don't have hyper parameters evaluated yet)
  // WE can start Template evolution with default hyper parameters because evaluation of hyper parameters on first iteration contributes only
  // to the upcoming evolutions as for the first iteration we have nothing to compare against for HyperParameter dimension //TODO make it configurable strategy
  //But it makes sense to search for HPs first. It is faster and brings more value
  override def getBestFromHallOfFame: HyperParametersField = hallOfFame.headOption.map(_.field).getOrElse{getInitialPopulation.individuals.randElement}

  override def getBestFromPopulation(workingDF: DataFrame): EvaluatedHyperParametersField = {
    debug("Getting best individual from population...")
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

  override def toString: String = modelsHParameterGroups.map(group => group.hpParameters.map(parameter => s"$parameter").mkString(" , ")).mkString(" | ")
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
trait DoubleHPRange extends HPRange[Double] {

  def round(value: Double, places: Int): Double = {
    if (places < 0) throw new IllegalArgumentException
    val bd = BigDecimal(value).setScale(places, RoundingMode.HALF_UP)
    bd.doubleValue
  }

  def getNextWithinTheRange: Double = {
    new Random().nextInt(max.toInt) + min
  }
}

sealed trait HParameter[+T] {
  def getDefault:T
}

trait MutableHParameter[+T, V <: MutableHParameter[T, V]] extends HParameter[T] {
  def currentValue: T
  def mutate(): V
}


class HPPopulation(val individuals: Seq[ HyperParametersField]) extends Population[HyperParametersField]

case class EvaluatedHyperParametersField(field: HyperParametersField, score:Double) extends Evaluated[EvaluatedHyperParametersField] {
  override type ItemType = HyperParametersField
  override type FitnessType = Double

  override type ParamsType = AnyVal //Unused

  override def params:ParamsType = 42

  override def item: HyperParametersField = field
  override def result: Double = score

  override def compare(other: EvaluatedHyperParametersField): Boolean = score > other.score // TODO bigger the better
}

object EvaluatedHyperParametersField {
  implicit val individualsOrdering: Ordering[EvaluatedHyperParametersField] = new Ordering[EvaluatedHyperParametersField] {
    override def compare(x: EvaluatedHyperParametersField, y: EvaluatedHyperParametersField): Int = {
      x.score.compareTo(y.score) // TODO swap y and x depending on what is needed in the Priority queue
    }
  }
}
package com.automl

import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.helper.{FitnessResult, PopulationHelper, TemplateTreeHelper}
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.{TemplateMember, TemplateTree}
import com.typesafe.scalalogging.LazyLogging
import kamon.Kamon

case class EvaluatedTemplateData(id: String,
                                 template: TemplateTree[TemplateMember],
                                 algorithm: TemplateTree[TemplateMember],
                                 fitness: FitnessResult,
                                 rank: Long = -1,
                                 probability: Double = -1,
                                 neighbours: Seq[EvaluatedTemplateData] = Nil,
                                 hyperParamsField: Option[HyperParametersField] = None) extends Evaluated[EvaluatedTemplateData] {

  type ItemType = TemplateTree[TemplateMember]
  type FitnessType = FitnessResult
  type ParamsType = HyperParametersField

  def idShort = s"$id:${template.member.name.take(5)}" // TODO add short name to members

  override def item: TemplateTree[TemplateMember] = template
  override def result: FitnessResult = fitness
  override def params: Option[HyperParametersField] = hyperParamsField

  def betterThan(that:EvaluatedTemplateData): Boolean = {
    if (theBiggerTheBetter(fitness.problemType))
      compare(that) > 0
    else compare(that) < 0
  }
  override def compare(that: EvaluatedTemplateData): Int = {
    val comparisonResult = fitness.compareTo(that.fitness)
    if (theBiggerTheBetter(fitness.problemType)) comparisonResult else -comparisonResult
  }

  def withRank(value: Long): EvaluatedTemplateData = copy(rank = value)
  def withProbability(value: Double): EvaluatedTemplateData = copy(probability = value)

  def renderPredictionsAsRow: String = idShort + ":" + fitness.dfWithPredictions.select("prediction").collect().map(_.getDouble(0)).mkString(" , ")

  // TODO or maybe keep metric in template,
  // because actually fitness of algorithm represents template. Maybe it is the only way to keep numeration consistent
  val kamonMetric = Kamon.gauge(s"kamon.automl.population.individual.$id")

  def sendMetric() = {
    val newValue = (fitness.getCorrespondingMetric * 1000000).toLong
    kamonMetric.set(newValue)
  }

  //Should be faster as we compare only certain fiels of the case class
  override def equals(obj: Any): Boolean = {
    require(obj.isInstanceOf[EvaluatedTemplateData])
    val another = obj.asInstanceOf[EvaluatedTemplateData]
    this.template == another.template && this.fitness.getCorrespondingMetric == another.fitness.getCorrespondingMetric
  }
}

object EvaluatedTemplateData extends LazyLogging {

//  implicit def individualsOrdering(theBiggerTheBetter: Boolean) = new Ordering[EvaluatedTemplateData] {
//    override def compare(x: EvaluatedTemplateData, y: EvaluatedTemplateData) = {
//      x.fitness.compareTo(y.fitness)
//    }
//  }

  implicit def evaluatedHelper(individual: EvaluatedTemplateData) = new {
    def render(problemType: ProblemType): String = {
        s"${TemplateTreeHelper.renderAsString_v2(individual.template)} Score: ${individual.fitness.toString}"
    }
  }
}

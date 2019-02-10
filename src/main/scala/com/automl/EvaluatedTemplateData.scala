package com.automl

import com.automl.helper.{FitnessResult, PopulationHelper}
import com.automl.template.{TemplateMember, TemplateTree}
import com.typesafe.scalalogging.LazyLogging
import kamon.Kamon

case class EvaluatedTemplateData(id: String,
                                 template: TemplateTree[TemplateMember],
                                 algorithm: TemplateTree[TemplateMember],
                                 fitness: FitnessResult,
                                 rank: Long = -1,
                                 probability: Double = -1,
                                 neighbours: Seq[EvaluatedTemplateData] = Nil) extends Evaluated[EvaluatedTemplateData] {

  override type ItemType = TemplateTree[TemplateMember]
  type FitnessType = FitnessResult

  def idShort = s"$id:${template.member.name.take(5)}" // TODO add short name to members


  override def item: TemplateTree[TemplateMember] = template
  override def result: FitnessResult = fitness


  override def compare(other: EvaluatedTemplateData): Boolean = fitness.orderTo(other.fitness)

  def withRank(value: Long): EvaluatedTemplateData = copy(rank = value)
  def withProbability(value: Double): EvaluatedTemplateData = copy(probability = value)

  // TODO or maybe keep metric in template,
  // because actually fitness of algorithm represents template. Maybe it is the only way to keep numeration consistent
  val kamonMetric = Kamon.gauge(s"kamon.automl.population.individual.$id")

  def sendMetric() = {
    val newValue = (fitness.getCorrespondingMetric * 1000000).toLong
    kamonMetric.set(newValue)
  }
}

object EvaluatedTemplateData extends LazyLogging {

  implicit val individualsOrdering = new Ordering[EvaluatedTemplateData] {
    override def compare(x: EvaluatedTemplateData, y: EvaluatedTemplateData) = {
      x.fitness.compareTo(y.fitness)
    }
  }

  implicit def individualHelper(individuals: Seq[EvaluatedTemplateData]) = new {

    // TODO No usage
    def printSortedByFitness(): Unit = {
      logger.debug(PopulationHelper.renderEvaluatedIndividuals(individuals))
    }
  }
}

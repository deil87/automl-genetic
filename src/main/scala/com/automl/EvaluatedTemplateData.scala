package com.automl

import com.automl.helper.FitnessResult
import com.automl.template.{TemplateMember, TemplateTree}
import com.typesafe.scalalogging.LazyLogging
import kamon.Kamon

case class EvaluatedTemplateData(id: String,
                                 template: TemplateTree[TemplateMember],
                                 algorithm: TemplateTree[TemplateMember],
                                 fitness: FitnessResult,
                                 rank: Long = -1,
                                 probability: Double = -1) {

  def withRank(value: Long): EvaluatedTemplateData = copy(rank = value)
  def withProbability(value: Double): EvaluatedTemplateData = copy(probability = value)

  // TODO or maybe keep metric in template,
  // because actually fitness of algorithm represents template. Maybe it is the only way to keep numeration consistent
  val kamonMetric = Kamon.gauge(s"kamon.automl.population.individual.$id")
  def sendMetric() = {
    val newValue = (fitness.fitnessError * 1000000).toLong
    kamonMetric.set(newValue)
  }
}

object EvaluatedTemplateData extends LazyLogging {

  implicit val individualsOrdering = new Ordering[EvaluatedTemplateData] {
    override def compare(x: EvaluatedTemplateData, y: EvaluatedTemplateData) = {
      if(x.fitness.fitnessError < y.fitness.fitnessError) 1
      else if(x.fitness.fitnessError > y.fitness.fitnessError) -1
      else 0
    }
  }

  implicit def individualHelper(individuals: Seq[EvaluatedTemplateData]) = new {

    def printSortedByFitness(): Unit = {
      individuals.zipWithIndex.sortBy(_._1.fitness.fitnessError).map { case (indivData, idx) =>
        (idx, s"$idx) ${indivData.fitness.fitnessError} \n")
      }.sortBy(_._1).foreach { case (_, str) => logger.info(str) }
    }
  }
}

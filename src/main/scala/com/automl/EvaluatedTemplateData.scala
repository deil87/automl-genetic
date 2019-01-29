package com.automl

import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem}
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

    def printSortedByFitness(): Unit = {
      individuals.zipWithIndex.sortBy(_._1.fitness.getCorrespondingMetric).map { case (indivData, idx) =>
        (idx, s"$idx) ${indivData.fitness.metricsMap} \n")
      }.sortBy(_._1).foreach { case (_, str) => logger.info(str) }
    }
  }
}

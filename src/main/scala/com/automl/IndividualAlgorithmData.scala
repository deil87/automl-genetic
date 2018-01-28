package com.automl

import com.automl.helper.FitnessResult
import com.automl.template.{TemplateMember, TemplateTree}
import kamon.Kamon

case class IndividualAlgorithmData(id: String,
                                   template: TemplateTree[TemplateMember],
                                   algorithm: TemplateTree[TemplateMember],
                                   fitness: FitnessResult,
                                   rank: Long = -1,
                                   probability: Double = -1) {

  def withRank(value: Long): IndividualAlgorithmData = copy(rank = value)
  def withProbability(value: Double): IndividualAlgorithmData = copy(probability = value)

  // TODO or maybe keep metric in template,
  // because actually fitness of algorithm represents template. Maybe it is the only way to keep numeration consistent
  val kamonMetric = Kamon.gauge(s"kamon.automl.population.individual.$id")
  def sendMetric() = {
    val newValue = (fitness.fitnessError * 1000000).toLong
    kamonMetric.set(newValue)
  }
}

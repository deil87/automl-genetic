package com.automl

import com.automl.helper.{FitnessResult, TemplateTreeHelper}
import com.automl.template.{TemplateMember, TemplateTree}
import com.typesafe.scalalogging.LazyLogging
import kamon.Kamon
import kamon.metric.CounterMetric
import org.apache.spark.ml.param.Params
import org.apache.spark.sql.DataFrame

import scala.collection.mutable

object PopulationEvaluator extends LazyLogging{

  private val cacheHitsCounterKamon = Kamon.counter("kamon.automl.cache_hits")

  def evaluateIndividuals(population: Population,
                          workingDataSet: DataFrame,
                          hyperParamsMap: Map[String, Seq[Params]] = Map.empty)
                         (implicit cache: mutable.Map[(TemplateTree[TemplateMember], Long), FitnessResult]): Seq[EvaluatedTemplateData] = {

    //TODO make use of hyperParamsMap for templated/nodes/classifiers

    population.individuals.zipWithIndex
      .map { case (individualTemplate, idx) => (idx, individualTemplate, TemplateTreeHelper.materialize(individualTemplate)) }
      .map { case (idx, template, materializedTemplate) =>

        val cacheKey = (materializedTemplate, workingDataSet.count())
        if (cache.isDefinedAt(cacheKey)) {
          logger.info(s"Cache hit happened for $idx-th individual based on: \n template: $template \n algorithm: $materializedTemplate \n")
          cacheHitsCounterKamon.increment(1)
        }
        val fr = cache.getOrElseUpdate(cacheKey, {
          logger.info(s"Calculated new value for $idx-th individual based on: \n template: $template \n algorithm: $materializedTemplate \n")
          // TODO can we split it randomly here???

          val Array(trainingSplit, testSplit) = workingDataSet.randomSplit(Array(0.67, 0.33), 11L)
          materializedTemplate.evaluateFitness(trainingSplit, testSplit)
        })
        val iad = EvaluatedTemplateData(idx.toString, template, materializedTemplate, fr)
        iad.sendMetric()
        iad
      }
  }
}

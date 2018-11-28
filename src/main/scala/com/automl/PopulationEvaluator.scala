package com.automl

import java.util.concurrent.TimeUnit

import akka.actor.{ActorRef, ActorSystem}
import akka.util.Timeout
import com.automl.helper.{FitnessResult, TemplateTreeHelper}
import com.automl.route.UpdateWeb
import com.automl.template.{TemplateMember, TemplateTree}
import com.typesafe.scalalogging.LazyLogging
import kamon.Kamon
import kamon.metric.CounterMetric
import org.apache.spark.ml.param.Params
import org.apache.spark.sql.DataFrame

import scala.collection.mutable
import scala.concurrent.Future
import scala.concurrent.duration.FiniteDuration
import scala.util.{Failure, Success}

class PopulationEvaluator(implicit as: ActorSystem) extends LazyLogging{

  private val cacheHitsCounterKamon = Kamon.counter("kamon.automl.cache_hits")

  import scala.concurrent.ExecutionContext.Implicits.global
  implicit val timeout = Timeout(FiniteDuration(1, TimeUnit.SECONDS))
  val webClientNotifier: Future[ActorRef] = as.actorSelection("user/" + "webClientNotifier").resolveOne()/*.onComplete {
    case Success(actorRef) => // logic with the actorRef
    case Failure(ex) => //
  }*/

  def evaluateIndividuals(population: Population,
                          workingDataSet: DataFrame,
                          hyperParamsMap: Map[String, Seq[Params]] = Map.empty)
                         (implicit cache: mutable.Map[(TemplateTree[TemplateMember], Long), FitnessResult]): Seq[EvaluatedTemplateData] = {




    //TODO make use of hyperParamsMap for templated/nodes/classifiers

    population.individuals.zipWithIndex
      .map { case (individualTemplate, idx) =>

        // TODO we don't use Wildcards and therefore no need in materialization. Should we use them ? It could be a variance regulator.
        val materializedTemplate = TemplateTreeHelper.materialize(individualTemplate)

        val cacheKey = (materializedTemplate, workingDataSet.count())
        if (cache.isDefinedAt(cacheKey)) {
          logger.debug(s"Cache hit happened for $idx-th individual based on: \n template: $individualTemplate \n algorithm: $materializedTemplate \n")
          cacheHitsCounterKamon.increment(1)
        }
        val fr: FitnessResult = cache.getOrElseUpdate(cacheKey, {
          logger.debug(s"Calculated new value for $idx-th individual based on: \n template: $individualTemplate \n algorithm: $materializedTemplate \n")
          // TODO can we split it randomly here???

          val Array(trainingSplit, testSplit) = workingDataSet.randomSplit(Array(0.67, 0.33), 11L)
          materializedTemplate.evaluateFitness(trainingSplit, testSplit)
        })
        webClientNotifier.map(wcn => wcn ! UpdateWeb(s"Evaluated ${TemplateTreeHelper.print2(materializedTemplate)} with fitness value: " + fr.fitnessError))
        val iad = EvaluatedTemplateData(idx.toString, individualTemplate, materializedTemplate, fr)
        iad.sendMetric()
        iad
      }
  }
}

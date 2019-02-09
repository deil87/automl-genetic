package com.automl.evolution.evaluation

import java.util.concurrent.TimeUnit

import akka.actor.{ActorRef, ActorSystem}
import akka.util.Timeout
import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.helper.{FitnessResult, TemplateTreeHelper}
import com.automl.problemtype.ProblemType
import com.automl.route.UpdateWeb
import com.automl.template.{TemplateMember, TemplateTree}
import com.automl.{EvaluatedTemplateData, TPopulation}
import com.typesafe.scalalogging.LazyLogging
import kamon.Kamon
import org.apache.spark.sql.DataFrame

import scala.collection.mutable
import scala.concurrent.Future
import scala.concurrent.duration.FiniteDuration

class TemplateSimpleEvaluator(implicit as: ActorSystem) extends PopulationEvaluator[TPopulation] with LazyLogging{

  private val cacheHitsCounterKamon = Kamon.counter("kamon.automl.cache_hits")

  import scala.concurrent.ExecutionContext.Implicits.global
  implicit val timeout = Timeout(FiniteDuration(1, TimeUnit.SECONDS))
  val webClientNotifier: Future[ActorRef] = as.actorSelection("user/" + "webClientNotifier").resolveOne()/*.onComplete {
    case Success(actorRef) => // logic with the actorRef
    case Failure(ex) => //
  }*/


  override def evaluateIndividuals(population: TPopulation,
                                   workingDataSet: DataFrame,
                                   hyperParamsMap: HyperParametersField,
                                   problemType: ProblemType)
                                  (implicit cache: mutable.Map[(TemplateTree[TemplateMember], Long), FitnessResult]): Seq[EvaluatedTemplateData] = {

    //TODO make use of hyperParamsMap for templated/nodes/classifiers

    population.individuals.zipWithIndex
      .map { case (individualTemplate, idx) =>

        // TODO we don't use Wildcards and therefore no need in materialization. Should we use them ? It could be a variance regulator.
        val materializedTemplate = TemplateTreeHelper.materialize(individualTemplate)

        val cacheKey = (materializedTemplate, workingDataSet.count())
        if (cache.isDefinedAt(cacheKey)) {
//          println(s"${Console.BLUE} Cache hit happened for $idx-th individual based on: \n template: $individualTemplate \n algorithm: $materializedTemplate \n")
          logger.debug(s"Cache hit happened for $idx-th individual based on: \n template: $individualTemplate \n algorithm: $materializedTemplate \n")
          cacheHitsCounterKamon.increment(1)
        }
        val fr: FitnessResult = cache.getOrElseUpdate(cacheKey, {
          logger.debug(s"Calculated new value for $idx-th individual based on template: $individualTemplate \n algorithm: $materializedTemplate \n")
          // TODO can we split it randomly here???

          val Array(trainingSplit, testSplit) = workingDataSet.randomSplit(Array(0.67, 0.33), 11L)
          materializedTemplate.evaluateFitness(trainingSplit, testSplit, problemType)
        })
        webClientNotifier.map(wcn => wcn ! UpdateWeb(s"Evaluated ${TemplateTreeHelper.renderAsString_v2(materializedTemplate)} with fitness value: " + fr.metricsMap))
        val iad = EvaluatedTemplateData(idx.toString + ":" + individualTemplate.id, individualTemplate, materializedTemplate, fr)
        iad.sendMetric()
        iad
      }
  }
}

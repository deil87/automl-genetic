package com.automl.evolution.evaluation

import java.util.concurrent.TimeUnit

import akka.actor.{ActorRef, ActorSelection, ActorSystem}
import akka.util.Timeout
import com.automl.evolution.dimension.TemplateEvolutionDimension
import com.automl.evolution.dimension.hparameter.{HyperParametersField, HyperParametersEvolutionDimension}
import com.automl.evolution.diversity.{CosineSimilarityAssistant, DistanceMetric, DistanceStrategy, MultidimensionalDistanceMetric}
import com.automl.helper.{FitnessResult, TemplateTreeHelper}
import com.automl.population.TPopulation
import com.automl.problemtype.ProblemType
import com.automl.route.{UpdateWeb, UpdateWebWithJson}
import com.automl.template.{TemplateMember, TemplateTree}
import com.automl.{ConfigProvider, ConsistencyChecker, EvaluatedTemplateData, PaddedLogging}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.sql.DataFrame
import utils.BenchmarkHelper

import scala.collection.mutable
import utils.SparkMLUtils._

import scala.concurrent.Future
import scala.concurrent.duration.FiniteDuration
import spray.json._
import DefaultJsonProtocol._



case class EvaluatedTemplateDataDTO(key: String, templateId: String, description: String, fitness: String) {
}

object EvaluatedTemplateDataDTO {
  def apply(evaluatedTemplateData: EvaluatedTemplateData, problemType: ProblemType): EvaluatedTemplateDataDTO = new EvaluatedTemplateDataDTO(
    "evaluatedTemplateData",
    evaluatedTemplateData.template.id,
    evaluatedTemplateData.render(problemType),
    evaluatedTemplateData.fitness.getCorrespondingMetric.toString
  )
}

object EvaluatedTemplateDataDTOJsonProtocol extends DefaultJsonProtocol {
  implicit val colorFormat = jsonFormat4(EvaluatedTemplateDataDTO.apply)
}


/**
  * Neighbourhoods are going to be found based on Phenotypic notion of the distance
  */
//TODO DistanceMetric type could be dependant/parametrized on ProblemType
class TemplateNSLCEvaluator[DistMetric <: MultidimensionalDistanceMetric](
                                                                           templateEvDimension: TemplateEvolutionDimension,
                                                                           hyperParamsEvDimOpt: Option[HyperParametersEvolutionDimension])
                                                                         (implicit as: ActorSystem, val logPaddingSize: Int)
  extends PopulationEvaluator[TPopulation, TemplateTree[TemplateMember], EvaluatedTemplateData] with ConsistencyChecker with PaddedLogging{


  override type CacheKeyType = (TemplateTree[TemplateMember], Option[HyperParametersField], Long)

  val tdConfig = ConfigProvider.config.getConfig("evolution.templateDimension")

  lazy val testSplitRatio: Double = tdConfig.getDouble("testSplitRatio")

  // Updating UI
  val webClientNotifier: ActorSelection = as.actorSelection("user/webClientNotifier")

  override def evaluateIndividuals(population: TPopulation,
                                   workingDF: DataFrame,
                                   problemType: ProblemType, // TODO could be removed as we can use templateEvDimension.problemType
                                   evaluationContextInfo: EvaluationContextInfo,
                                   seed: Long)
                                  (implicit cache: mutable.Map[(TemplateTree[TemplateMember], Option[HyperParametersField], Long), FitnessResult]): Seq[EvaluatedTemplateData] = {
    debug("TemplateNSLCEvaluator. Evaluation of templates have started.")

    val bestHPFieldFromCoevolution: Option[HyperParametersField] = hyperParamsEvDimOpt.map{ hyperParamsEvDim =>
      /* Template dimension depends on others dimensions and we need to get data from them first.
      This could be implemented in a custom hardcoded evaluator or with dependencies tree */
      //TODO  For how long we want to search for a hyperparameters? We can introduce HPSearchStepsPerGeneration parameter or we need to add logic that decides how often we need to evolve subdimensions
      debug("Before evaluation of Template population we want to get best individuals from coevolutions we depend on. Checking HP coevolution...")
      if (hyperParamsEvDim.hallOfFame.isEmpty || hyperParamsEvDim.currentWorkingDFSize != workingDF.count()) {
        debug("HP coevolution is either was not yet evaluated before or need update due to increased size of workingDF")
        hyperParamsEvDim.evolveFromLastPopulation(workingDF, ???) // TODO consider stratified sample for first iteration or maybe for all iterations
      }
      val bestHPField = hyperParamsEvDim.getBestFromHallOfFame
      debug(s"Evaluation of hyper parameter coevolution completed. \nBest: ${bestHPField.toString} \nProceeding to templates evaluations...")
      bestHPField
    }

    //TODO make use of hyperParamsMap for templates/nodes/classifiers

    val Array(trainingSplit, testSplit) = workingDF.randomSplit(Array(1 - testSplitRatio, testSplitRatio), seed)
    trainingSplit.cache()
    testSplit.cache()

    val evaluatedTemplatesData: Seq[EvaluatedTemplateData] = population.individuals.zipWithIndex
      .map { case (individualTemplate, idx) =>

        // TODO we don't use Wildcards and therefore no need in materialization. Should we use them ? It could be a variance regulator.
        val materializedTemplate = TemplateTreeHelper.materialize(individualTemplate)

        val cacheKey = generateCacheKey(workingDF, bestHPFieldFromCoevolution, individualTemplate)
        if (cache.isDefinedAt(cacheKey)) {
          debug(s"Cache hit happened for $idx-th individual based on: template: $cacheKey")
          val keyHashCode = cacheKey.hashCode()
          val restoredFromCache = cache(cacheKey)
          debug(s"Retrieved value from the cache with hashCode = ${keyHashCode} : ${restoredFromCache}")
        }

        //TODO FIX We are storing this result in cache but not in the priority queue
        val fitness: FitnessResult = cache.getOrElseUpdate(cacheKey, {

          individualTemplate.setLogPadding(logPaddingSize)
          materializedTemplate.setLogPadding(logPaddingSize)

          val fitnessResult = individualTemplate.evaluateFitness(trainingSplit, testSplit, problemType, bestHPFieldFromCoevolution)
          debug(s"Entry $cacheKey with hashCode = ${cacheKey.hashCode()} was added to the cache with score = $fitnessResult")
          fitnessResult
        })

        val result = EvaluatedTemplateData(idx.toString + ":" + individualTemplate.id, individualTemplate,
          materializedTemplate, fitness, hyperParamsFieldFromCoevolution = bestHPFieldFromCoevolution)
        result.setEvaluationContextInfo(evaluationContextInfo)

        templateEvDimension.hallOfFame += result

        import EvaluatedTemplateDataDTOJsonProtocol._
        import spray.json._
        val evaluatedTemplateDataDTO = EvaluatedTemplateDataDTO(result, problemType).toJson
        webClientNotifier ! UpdateWebWithJson(evaluatedTemplateDataDTO.prettyPrint)
        result
      }
    evaluatedTemplatesData
  }

  private def generateCacheKey(workingDF: DataFrame, bestHPFieldFromCoevolution: Option[HyperParametersField], individualTemplate: TemplateTree[TemplateMember]) = {
    if(bestHPFieldFromCoevolution.isDefined) // TODO it would be great to have concise representation for Template to use it as a hashcode.
      (individualTemplate, bestHPFieldFromCoevolution, workingDF.count())
    else
      (individualTemplate, None/*individualTemplate.member.hpGroup*/, workingDF.count())
  }
}


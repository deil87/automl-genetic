package com.automl.evolution.evaluation

import java.util.concurrent.TimeUnit

import akka.actor.{ActorRef, ActorSelection, ActorSystem}
import akka.util.Timeout
import com.automl.evolution.dimension.TemplateEvolutionDimension
import com.automl.evolution.dimension.hparameter.{HyperParametersEvolutionDimension, HyperParametersField}
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
import com.typesafe.config.Config
import org.apache.spark.mllib.util.MLUtils



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


  override type CacheKeyType = (TemplateTree[TemplateMember], Option[HyperParametersField], Long, Int)

  val tdConfig: Config = ConfigProvider.config.getConfig("evolution.templateDimension")

  lazy val testSplitRatio: Double = tdConfig.getDouble("testSplitRatio")

  lazy val globalCVNumFolds: Int = tdConfig.getInt("globalCVNumFolds")

  // Updating UI
  val webClientNotifier: ActorSelection = as.actorSelection("user/webClientNotifier")

  override def evaluateIndividuals(population: TPopulation,
                                   workingDF: DataFrame,
                                   problemType: ProblemType, // TODO could be removed as we can use templateEvDimension.problemType
                                   evaluationContextInfo: EvaluationContextInfo,
                                   seed: Long)
                                  (implicit cache: mutable.Map[CacheKeyType, FitnessResult]): Seq[EvaluatedTemplateData] = {
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

    val trainTestPairs: Array[(DataFrame, DataFrame)] = workingDF.toTrainTestPairs(globalCVNumFolds, seed)

    testSplitRatio // TODO deprecated?

    def bestResultSoFar = templateEvDimension.hallOfFame.headOption.map(_.fitness)

    val evaluatedTemplatesData: Seq[EvaluatedTemplateData] = population.individuals.zipWithIndex
      .map { case (individualTemplate, idx) =>


        // TODO we don't use Wildcards and therefore no need in materialization. Should we use them ? It could be a variance regulator.
        val materializedTemplate = TemplateTreeHelper.materialize(individualTemplate)

        import util.control.Breaks._
        val evaluatedCV = mutable.Buffer.empty[FitnessResult]
        breakable {
          var pairIdx = 0
          for( (trainSplit, testSplit) <- trainTestPairs) {
            debug(s"Evaluating ${pairIdx}th fold:")
//            trainSplit.showN_AndContinue(trainSplit.count().toInt, s"train from ${pairIdx}th fold:")
//            testSplit.showN_AndContinue(trainSplit.count().toInt, s"test from ${pairIdx}th fold:")
            val cacheKey = generateCacheKey(workingDF, testSplit, bestHPFieldFromCoevolution, individualTemplate)
            if (cache.isDefinedAt(cacheKey)) {
              debug(s"Cache hit happened for $idx-th individual based on: template: $cacheKey")
              val keyHashCode = cacheKey.hashCode()
              val restoredFromCache = cache(cacheKey)
              debug(s"Retrieved value from the cache with hashCode = ${keyHashCode} : ${restoredFromCache}")
            }

            //TODO FIX We are storing this result in cache but not in the priority queue
            val currentPairFitness: FitnessResult = cache.getOrElseUpdate(cacheKey, {

              individualTemplate.setLogPadding(logPaddingSize)
              materializedTemplate.setLogPadding(logPaddingSize)

              val fitnessResult = individualTemplate.evaluateFitness(trainSplit, testSplit, problemType, bestHPFieldFromCoevolution)
              debug(s"Entry $cacheKey with hashCode = ${cacheKey.hashCode()} was added to the cache with score = $fitnessResult")
              fitnessResult
            })
            evaluatedCV.append(currentPairFitness)
            bestResultSoFar.foreach { result =>
              if (result.betterThan(currentPairFitness, 2.0)) {
                info("!!!!! SKIPPING other folds as performance looks bad")
                break() // leaving one fold performance as a representative
              }
            } // TODO test
            pairIdx+=1
          }
        }

        // ugly
        val averagedFitness = evaluatedCV.reduceOption((fitnessLeft, fitnessRight) => {
          FitnessResult(
            sumMaps(fitnessLeft.metricsMap, fitnessRight.metricsMap),
            problemType = fitnessLeft.problemType,
            fitnessLeft.dfWithPredictions) // TODO we will need to store every fold's predictions. Use zip with index. This way test split from first fold will be stored in FitnessResult and therefore used in neighbours finder.
        }).map(fitnessResultSum => {
          FitnessResult(
            fitnessResultSum.metricsMap.map(item => (item._1, item._2 / evaluatedCV.size)),
            problemType = fitnessResultSum.problemType,
            fitnessResultSum.dfWithPredictions)
        }).get

        val result = EvaluatedTemplateData(idx.toString + ":" + individualTemplate.id, individualTemplate,
          materializedTemplate, averagedFitness, hyperParamsFieldFromCoevolution = bestHPFieldFromCoevolution)
        result.setEvaluationContextInfo(evaluationContextInfo)


        registerInHallOfFame(result)
        notifyWebClient(problemType, result)

        result
      }
    evaluatedTemplatesData
  }

  def sumMaps(mapLeft: Map[String,Double], mapRight: Map[String,Double]): Map[String, Double] = {
    mapLeft.map{ case (key, value) => (key, value + mapRight(key))}
  }

  private def notifyWebClient(problemType: ProblemType, result: EvaluatedTemplateData) = {
    import EvaluatedTemplateDataDTOJsonProtocol._
    import spray.json._
    val evaluatedTemplateDataDTO = EvaluatedTemplateDataDTO(result, problemType).toJson
    webClientNotifier ! UpdateWebWithJson(evaluatedTemplateDataDTO.prettyPrint)
  }

  private def registerInHallOfFame(result: EvaluatedTemplateData) = {
    templateEvDimension.hallOfFame += result
  }

  private def generateCacheKey(workingDF: DataFrame, testSplitDF: DataFrame, bestHPFieldFromCoevolution: Option[HyperParametersField], individualTemplate: TemplateTree[TemplateMember]) = {
    if(bestHPFieldFromCoevolution.isDefined) // TODO it would be great to have concise representation for Template to use it as a hashcode.
      (individualTemplate, bestHPFieldFromCoevolution, workingDF.count(), testSplitDF.hashCode())
    else
      (individualTemplate, None/*individualTemplate.member.hpGroup*/, workingDF.count(), testSplitDF.hashCode()) //TODO probably don't need second element in tuple as we check them from `individualTemplate`
  }
}


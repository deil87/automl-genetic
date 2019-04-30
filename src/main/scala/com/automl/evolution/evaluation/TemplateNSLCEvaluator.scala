package com.automl.evolution.evaluation

import akka.actor.ActorSystem
import com.automl.evolution.dimension.TemplateEvolutionDimension
import com.automl.evolution.dimension.hparameter.{HyperParametersField, TemplateHyperParametersEvolutionDimension}
import com.automl.evolution.diversity.{CosineSimilarityAssistant, DistanceMetric, DistanceStrategy, MultidimensionalDistanceMetric}
import com.automl.helper.{FitnessResult, TemplateTreeHelper}
import com.automl.population.TPopulation
import com.automl.problemtype.ProblemType
import com.automl.template.{TemplateMember, TemplateTree}
import com.automl.{ConsistencyChecker, EvaluatedTemplateData, PaddedLogging}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.sql.DataFrame
import utils.BenchmarkHelper

import scala.collection.mutable
import utils.SparkMLUtils._

/**
  * Neighbourhoods are going to be found based on Phenotypic notion of the distance
  */
//TODO DistanceMetric type could be dependant/parametrized on ProblemType
class TemplateNSLCEvaluator[DistMetric <: MultidimensionalDistanceMetric](
                                                                           templateEvDimension: TemplateEvolutionDimension,
                                                                           hyperParamsEvDim: TemplateHyperParametersEvolutionDimension)
                                                                         (implicit as: ActorSystem, val logPaddingSize: Int)
  extends PopulationEvaluator[TPopulation, TemplateTree[TemplateMember], EvaluatedTemplateData] with ConsistencyChecker with PaddedLogging{


  override type CacheKeyType = (TemplateTree[TemplateMember], HyperParametersField, Long)

  override def evaluateIndividuals(population: TPopulation,
                                   workingDF: DataFrame,
                                   problemType: ProblemType,
                                   seed: Long)
                                  (implicit cache: mutable.Map[(TemplateTree[TemplateMember], HyperParametersField, Long), FitnessResult]): Seq[EvaluatedTemplateData] = {

    debug("TemplateNSLCEvaluator. Evaluation of templates have started.")

    /* Template dimension depends on others dimensions and we need to get data from them first.
   This could be implemented in a custom hardcoded evaluator or with dependencies tree */
    //TODO  For how long we want to search for a hyperparameters? We can introduce HPSearchStepsPerGeneration parameter or we need to add logic that decides how often we need to evolve subdimensions
    debug("Before evaluation of Template population we want to get best individuals from coevolutions we depend on. Checking HP coevolution...")
    if(hyperParamsEvDim.hallOfFame.isEmpty || hyperParamsEvDim.currentWorkingDFSize!= workingDF.count()) {
      debug("HP coevolution is either was not yet evaluated before or need update due to increased size of workingDF")
      hyperParamsEvDim.evolveFromLastPopulation(workingDF) // TODO consider stratified sample for first iteration or maybe for all iterations
    }
    val bestHPField =  hyperParamsEvDim.getBestFromHallOfFame
    debug(s"Evaluation of hyper parameter coevolution completed. \nBest: ${bestHPField.toString} \nProceeding to templates evaluations...")

    //TODO make use of hyperParamsMap for templated/nodes/classifiers

    val Array(trainingSplit, testSplit) = workingDF.randomSplit(Array(0.67, 0.33), 11L)
    trainingSplit.cache()
    testSplit.cache()

    val evaluatedTemplatesData: Seq[EvaluatedTemplateData] = population.individuals.zipWithIndex
      .map { case (individualTemplate, idx) =>

        // TODO we don't use Wildcards and therefore no need in materialization. Should we use them ? It could be a variance regulator.
        val materializedTemplate = TemplateTreeHelper.materialize(individualTemplate)

        val cacheKey = (materializedTemplate, bestHPField, workingDF.count())
        if (cache.isDefinedAt(cacheKey)) {
          debug(s"Cache hit happened for $idx-th individual based on: template: $cacheKey")
          debug(s"Retrieved value from the cache with hashCode = ${cacheKey.hashCode()} : ${cache(cacheKey)}")
        }

        //TODO FIX We are storing this result in cache but not into the queue
        val fitness: FitnessResult = cache.getOrElseUpdate(cacheKey, {

          materializedTemplate.setLogPadding(logPaddingSize)
          val fitnessResult = materializedTemplate.evaluateFitness(trainingSplit, testSplit, problemType, bestHPField)
          debug(s"Entry $cacheKey with hashCode = ${cacheKey.hashCode()} was added to the cache with score = $fitnessResult")
          fitnessResult
        })
        val result = EvaluatedTemplateData(idx.toString + ":" + materializedTemplate.id, materializedTemplate, materializedTemplate, fitness, hyperParamsField = bestHPField)
        templateEvDimension.hallOfFame += result
        result
      }
    evaluatedTemplatesData
  }
}

package com.automl.evolution.evaluation

import akka.actor.ActorSystem
import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.evolution.diversity.{CosineSimilarityAssistant, DistanceMetric, DistanceStrategy, MultidimensionalDistanceMetric}
import com.automl.helper.{FitnessResult, TemplateTreeHelper}
import com.automl.problemtype.ProblemType
import com.automl.template.{TemplateMember, TemplateTree}
import com.automl.{EvaluatedTemplateData, TPopulation}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.sql.DataFrame
import utils.BenchmarkHelper

import scala.collection.mutable

/**
  * Neighbourhoods are going to be found based on Phenotypic notion of the distance
  */
//TODO DistanceMetric type could be dependant/parametrized on ProblemType
class TemplateNSLCEvaluator[DistMetric <: MultidimensionalDistanceMetric]( distanceStrategy: DistanceStrategy[DataFrame, DistMetric])(implicit as: ActorSystem)
  extends PopulationEvaluator[TPopulation] with LazyLogging{



  override def evaluateIndividuals(population: TPopulation,
                                   workingDataSet: DataFrame,
                                   hyperParamsField: HyperParametersField,
                                   problemType: ProblemType)
                                  (implicit cache: mutable.Map[(TemplateTree[TemplateMember], HyperParametersField, Long), FitnessResult]): Seq[EvaluatedTemplateData] = {

    //TODO make use of hyperParamsMap for templated/nodes/classifiers

    val evaluatedTemplateData = population.individuals.zipWithIndex
      .map { case (individualTemplate, idx) =>

        // TODO we don't use Wildcards and therefore no need in materialization. Should we use them ? It could be a variance regulator.
        val materializedTemplate = TemplateTreeHelper.materialize(individualTemplate)

        val cacheKey = (materializedTemplate, hyperParamsField, workingDataSet.count())
        if (cache.isDefinedAt(cacheKey)) {
          logger.debug(s"Cache hit happened for $idx-th individual based on: template: $cacheKey")
          logger.debug(s"Retrieved value from the cache with hashCode = ${cacheKey.hashCode()} : ${cache(cacheKey)}")
        }

        //TODO FIX We are storing this result in cache but not into the queue
        val fitness: FitnessResult = cache.getOrElseUpdate(cacheKey, {
          // TODO can we split it randomly here???

          val Array(trainingSplit, testSplit) = workingDataSet.randomSplit(Array(0.67, 0.33), 11L)
          trainingSplit.cache()
          testSplit.cache()
          val fitnessResult = materializedTemplate.evaluateFitness(trainingSplit, testSplit, problemType, hyperParamsField)
          logger.debug(s"Entry $cacheKey with hashCode = ${cacheKey.hashCode()} was added to the cache with score = $fitnessResult")
          fitnessResult
        })
        EvaluatedTemplateData(idx.toString + ":" + materializedTemplate.id, materializedTemplate, materializedTemplate, fitness)
      }
    evaluatedTemplateData
  }

  def findNeighbours(forWhomWeWantToFindNeighbours: Seq[EvaluatedTemplateData], neighbourhood: Seq[EvaluatedTemplateData], populationSize: Int) = {
    // QUESTION original  and withOffspring populations will have different neighbourhood sizes. Should we pass different `populationSize` or only size of original population
      val sizeOfTheNeighbourhood = if(populationSize <= 30) 3 else populationSize / 10
    BenchmarkHelper.time("Calculation of neighbours") {
      logger.debug(s"Starting to calculate neighbours (size = $sizeOfTheNeighbourhood) for population (TemplateNSLCEvaluator):")
      val withNeighbours = forWhomWeWantToFindNeighbours.map { currentEvaluatedTemplate =>

      val distanceToNeighbours = mutable.Buffer[(EvaluatedTemplateData, DistMetric)]() //TODO we can sort on the way by putting into sorted heap
        for (possibleNeighbour <- neighbourhood.diff(Seq(currentEvaluatedTemplate))) {
          //First we need to get predictions for all individuals and only then we can calculate phenotypic distances between them

          val distanceToTheNeighbour = distanceStrategy.getDistance(currentEvaluatedTemplate.fitness.dfWithPredictionsOnly, possibleNeighbour.fitness.dfWithPredictionsOnly)

          distanceToNeighbours.append((possibleNeighbour, distanceToTheNeighbour))
        }
        logger.debug(f"Distances from ${currentEvaluatedTemplate.idShort} (fitness = ${currentEvaluatedTemplate.fitness.getCorrespondingMetric}) to: " +
          f" \n  ${distanceToNeighbours.map(neigh => f"\t\t\t\t\t${neigh._1.idShort}  is  ${neigh._2.getDistanceMetrics(0)}%-20s (fitness = ${neigh._1.fitness.getCorrespondingMetric}%-20s)").mkString("  \n  ")}")

        //TODO try to not calculate distances if we have duplicates.

        //        val similarityEtalon = Array(0, numberOfRous.toDouble) // 1-st element is for minimal distance, second one is for max number of rows where errors/matches on the same positions were done.
        //        val neighbours = distanceToNeighbours.map { case (possibleNeighbour, cosineComponentsToPossibleNeighbour) =>
        //          (possibleNeighbour, CosineSimilarityAssistant.cosineSimilarity(similarityEtalon, cosineComponentsToPossibleNeighbour.getDistanceMetrics))
        //        }.
        val neighbours = distanceToNeighbours
          .sortWith(_._2.getDistanceMetrics(0) < _._2.getDistanceMetrics(0)) //TODO optimize
          .take(sizeOfTheNeighbourhood.toInt)
          .map(_._1)
        logger.debug(f"Chosen neighbours for ${currentEvaluatedTemplate.idShort} :  ${neighbours.map(_.idShort).mkString(" , ")}")


        currentEvaluatedTemplate.copy(neighbours = neighbours)
      }
      logger.debug(s"Finished calculation of neighbours for population in TemplateNSLCEvaluator.")
      withNeighbours
    }

  }

}

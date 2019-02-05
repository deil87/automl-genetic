package com.automl.evolution.evaluation

import akka.actor.ActorSystem
import com.automl.evolution.dimension.HyperParametersField
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
                                   hyperParamsMap: HyperParametersField,
                                   problemType: ProblemType)
                                  (implicit cache: mutable.Map[(TemplateTree[TemplateMember], Long), FitnessResult]): Seq[EvaluatedTemplateData] = {

    val numberOfRous = workingDataSet.count()
    val sizeOfTheNeighbourhood = Math.min(numberOfRous / 10, 20)
    //TODO make use of hyperParamsMap for templated/nodes/classifiers

    val evaluatedTemplateData = population.individuals.zipWithIndex
      .map { case (individualTemplate, idx) =>

        // TODO we don't use Wildcards and therefore no need in materialization. Should we use them ? It could be a variance regulator.
        val materializedTemplate = TemplateTreeHelper.materialize(individualTemplate)

        val cacheKey = (materializedTemplate, workingDataSet.count())
        if (cache.isDefinedAt(cacheKey)) {
          logger.debug(s"Cache hit happened for $idx-th individual based on: template: $individualTemplate")
//          logger.debug(s"Cache hit happened for $idx-th individual based on: \n template: $individualTemplate \n algorithm: $materializedTemplate \n")
        }

        //TODO FIX We are storing this result in cache but not into the queue
        val fitness: FitnessResult = cache.getOrElseUpdate(cacheKey, {
          logger.debug(s"Calculated new value for $idx-th individual based on: \n template: $individualTemplate")
          // TODO can we split it randomly here???

          val Array(trainingSplit, testSplit) = workingDataSet.randomSplit(Array(0.67, 0.33), 11L)
          trainingSplit.cache()
          testSplit.cache()
          materializedTemplate.evaluateFitness(trainingSplit, testSplit, problemType)
        })
        EvaluatedTemplateData(idx.toString, individualTemplate, materializedTemplate, fitness)
      }
    //First we need to get predictions for all individuals and only then we can calculate phenotypic distances between them

    BenchmarkHelper.time("Calculation of neighbours :") {
      logger.debug(s"Starting to calculate neighbours for population in TemplateNSLCEvaluator:")
      val withNeighbours = evaluatedTemplateData.map { currentEvaluatedTemplate =>
        val distanceToNeighbours = mutable.Buffer[(EvaluatedTemplateData, DistMetric)]()
        for (possibleNeighbour <- evaluatedTemplateData.diff(Seq(currentEvaluatedTemplate))) {
          val distanceToTheNeighbour = distanceStrategy.getDistance(currentEvaluatedTemplate.fitness.dfWithPredictionsOnly, possibleNeighbour.fitness.dfWithPredictionsOnly)
          distanceToNeighbours.append((possibleNeighbour, distanceToTheNeighbour))
        }

//        val similarityEtalon = Array(0, numberOfRous.toDouble) // 1-st element is for minimal distance, second one is for max number of rows where errors/matches on the same positions were done.
        val similarityEtalon = Array(0.0) //TODO or just replace with sorting

        val neighbours = distanceToNeighbours.map { case (possibleNeighbour, cosineComponentsToPossibleNeighbour) =>
          (possibleNeighbour, CosineSimilarityAssistant.cosineSimilarity(similarityEtalon, cosineComponentsToPossibleNeighbour.getDistanceMetrics))
        }.
          sorted
          .take(sizeOfTheNeighbourhood.toInt)
          .map(_._1)

        currentEvaluatedTemplate.copy(neighbours = neighbours)
      }
      logger.debug(s"Finished calculation of neighbours for population in TemplateNSLCEvaluator.")
      withNeighbours
    }
  }

}

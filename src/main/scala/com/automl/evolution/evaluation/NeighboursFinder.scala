package com.automl.evolution.evaluation

import akka.actor.ActorSystem
import com.automl.evolution.dimension.TemplateEvolutionDimension
import com.automl.evolution.dimension.hparameter.{HyperParametersField, HyperParametersEvolutionDimension}
import com.automl.evolution.diversity.{DistanceStrategy, MultidimensionalDistanceMetric}
import com.automl.helper.{FitnessResult, TemplateTreeHelper}
import com.automl.problemtype.ProblemType
import com.automl.template.{TemplateMember, TemplateTree}
import com.automl.{ConsistencyChecker, EvaluatedTemplateData, PaddedLogging}
import org.apache.spark.sql.DataFrame
import utils.BenchmarkHelper
import utils.SparkMLUtils._

import scala.collection.mutable


class NeighboursFinder[DistMetric <: MultidimensionalDistanceMetric](distanceStrategy: DistanceStrategy[DataFrame, DistMetric])
                                                                    (implicit as: ActorSystem, val logPaddingSize: Int)
  extends ConsistencyChecker with PaddedLogging{

  def findNeighbours(forWhomWeWantToFindNeighbours: Seq[EvaluatedTemplateData], neighbourhood: Seq[EvaluatedTemplateData], populationSize: Int) = {
    // QUESTION original  and withOffspring populations will have different neighbourhood sizes. Should we pass different `populationSize` or only size of original population
    val sizeOfTheNeighbourhood = if (populationSize <= 30) 3 else populationSize / 10
    BenchmarkHelper.time("Calculation of neighbours"){ // Note: logPaddingSize is set below as third list of parameters
      checkOrderOfPredictionsIsTheSame(forWhomWeWantToFindNeighbours)

      forWhomWeWantToFindNeighbours.foreach { currentEvaluatedTemplate =>
//        compute most difficult instances. show them on stdout and show individuals that were added to population in the end of generation.
        debug(currentEvaluatedTemplate.renderPredictionsAsRow)
      }

      debug(s"Starting to calculate neighbours (size = $sizeOfTheNeighbourhood) for population (TemplateNSLCEvaluator):")
      val withNeighbours = forWhomWeWantToFindNeighbours.map { currentEvaluatedTemplate =>

        val distanceToNeighbours = mutable.Buffer[(EvaluatedTemplateData, DistMetric)]() //TODO we can sort on the way by putting into sorted heap
        for (possibleNeighbour <- neighbourhood.diff(Seq(currentEvaluatedTemplate))) {
          //First we need to get predictions for all individuals and only then we can calculate phenotypic distances between them

          val distanceToTheNeighbour = distanceStrategy.getDistance(currentEvaluatedTemplate.fitness.dfWithPredictionsOnly, possibleNeighbour.fitness.dfWithPredictionsOnly)

          distanceToNeighbours.append((possibleNeighbour, distanceToTheNeighbour))
        }
        debug(f"Distances from ${currentEvaluatedTemplate.idShort} (fitness = ${currentEvaluatedTemplate.fitness.getCorrespondingMetric}) to: " +
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
        debug(f"Chosen neighbours for ${currentEvaluatedTemplate.idShort} :  ${neighbours.map(_.idShort).mkString(" , ")}")


        currentEvaluatedTemplate.copy(neighbours = neighbours)
      }
      debug(s"Finished calculation of neighbours for population in TemplateNSLCEvaluator.")
      withNeighbours
    }(logPaddingSize = logPaddingSize - 4)

  }

  private def checkOrderOfPredictionsIsTheSame(forWhomWeWantToFindNeighbours: Seq[EvaluatedTemplateData]) = {
    consistencyCheck {
      val firstIds = forWhomWeWantToFindNeighbours.map { evTemplate => evTemplate.fitness.dfWithPredictions.select("uniqueIdColumn").take(1).map(_.getLong(0)).head }
      if (!firstIds.forall(_ == firstIds.head)) {
        forWhomWeWantToFindNeighbours.foreach{etd =>
          val df = etd.fitness.dfWithPredictions.select("uniqueIdColumn","prediction")
          debug("Count of rows in prediction column" + df.count())
          debug(etd.template.render)
//          df.showN_AndContinue(20)
        }
        throw new IllegalStateException("Order of uniqueIdColumn is different in predictions")
      }
    }
  }
}

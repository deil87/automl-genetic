package com.automl.evolution.selection

import com.automl.EvaluatedTemplateData
import com.automl.helper.LinearRankingProbabilityStrategy
import com.typesafe.scalalogging.LazyLogging

import scala.util.Random

class RankSelectionStrategy extends LazyLogging{

  def parentSelection(selectionShare: Double, individuals: Seq[EvaluatedTemplateData]): Seq[EvaluatedTemplateData] = {
    require(selectionShare < 1 && selectionShare > 0, "Selection share parameter shoud be in range (0, 1)" )
    val numberOfCompetitors = individuals.length
    val numberOfParents = (numberOfCompetitors * selectionShare).toInt

    parentSelectionBySize(numberOfParents, individuals)
  }

  def parentSelectionBySize(numberOfParents: Int, individuals: Seq[EvaluatedTemplateData]): Seq[EvaluatedTemplateData] = {
    val orderedByFitness = individuals.sortWith(_.fitness.fitnessError > _.fitness.fitnessError)

    val numberOfCompetitors = individuals.length
    val linearRankingProbabilityStrategy = new LinearRankingProbabilityStrategy(numberOfCompetitors, parameter_S = 1.5)

    val ranked: Seq[EvaluatedTemplateData] = orderedByFitness
      .zipWithIndex
      .map { case (ind, rank) =>
        ind.withRank(rank).withProbability(linearRankingProbabilityStrategy.computeProbabilityFor(rank))
      }

    val rankedWithCumulativeProbs = ranked.drop(1).scanLeft(ranked.head){ case (acc, indd2) => indd2.copy(probability = indd2.probability + acc.probability)}
    logger.debug(rankedWithCumulativeProbs.map(r => (r.rank, r.probability)).mkString("\n"))

    var currentParentIndex = 0
    val selectedParents = new Array[EvaluatedTemplateData](numberOfParents)
    while (currentParentIndex < numberOfParents) {
      val r = Random.nextDouble()
      val rouletteWheel = rankedWithCumulativeProbs.dropWhile(individualData => individualData.probability < r)
      val selected = rouletteWheel.headOption.getOrElse(rankedWithCumulativeProbs.last)
      logger.debug(s"Selected for r = $r : $selected")
      selectedParents(currentParentIndex) = selected
      currentParentIndex += 1
    }
    selectedParents // TODO choose optimal data structure. Implicit conversion here.
  }

  def selectByProbability() = {

  }
}

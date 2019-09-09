package com.automl.evolution.selection

import com.automl.{EvaluatedTemplateData, PaddedLogging}
import com.automl.helper.{LinearRankingProbabilityStrategy, TemplateTreeHelper}
import com.typesafe.scalalogging.LazyLogging

import scala.collection.immutable
import scala.util.Random

// TODO make selection strategy generic and suitable for different coevolutions
class RankSelectionStrategy()(implicit val logPaddingSize: Int = 0) extends PaddedLogging{

  def parentSelectionByShare(selectionShare: Double, individuals: Seq[EvaluatedTemplateData]): Seq[EvaluatedTemplateData] = {
    val numberOfParents: Int = ratioToSIze(selectionShare, individuals)

    parentSelectionBySize(numberOfParents, individuals)
  }

  def selectionBySizeWithLocalCompetitions(numberOfParentsToSelect: Int, individuals: Seq[EvaluatedTemplateData]): Seq[EvaluatedTemplateData] = {

    debug(s"Selection ( task is to select $numberOfParentsToSelect out of ${individuals.length}):")
    val localScores = individuals.map{ individual =>
//      import collection.breakOut
//      val value = individual.neighbours.groupBy(_.fitness).map(_._2.head)//(breakOut)
      val value = individual.neighbours.filterNot(_.fitness == individual.fitness)
      val countOFNeighboursThatAreWorse = value.count(_.fitness.filterFun(individual.fitness))
      (individual, countOFNeighboursThatAreWorse)
    }

    // To display whole list in asc order
    val sortedBasedOnLocalScores = localScores.sortWith(_._2 < _._2)
    debug("Local competitions performances: " + sortedBasedOnLocalScores.map{case (template, score) => template.idShort + " has beaten -> " + score + " neighbours"}.mkString("\n\t\t\t\t\t", "\n\t\t\t\t\t", ""))

    val orderedBasedOnFitnessAgainstNeighbours: Seq[EvaluatedTemplateData] = sortedBasedOnLocalScores.map(_._1)

    val (selectedBySize, _) = selectNIndividualsFromSortedByRankCollectionWithoutReplacement(numberOfParentsToSelect, orderedBasedOnFitnessAgainstNeighbours)
    selectedBySize
  }

  def selectNIndividualsFromSortedByRankCollectionWithoutReplacement(numberOfParentsToSelect: Int, orderedBasedOnFitnessAgainstNeighbours: Seq[EvaluatedTemplateData]) = {
    (0 until numberOfParentsToSelect).foldLeft((Seq[EvaluatedTemplateData](), orderedBasedOnFitnessAgainstNeighbours)) { case ((selectedAcc, rest), next) => {
      val evaluatedTemplateDataWithAssignedProbs = new RankBasedSelectionProbabilityAssigner[EvaluatedTemplateData].assign(rest.toList)
      debug(s"Selecting by rank based probabilities from [ ${evaluatedTemplateDataWithAssignedProbs.map { case (individual, assignedProb) => s"${individual.idShort} ${TemplateTreeHelper.renderAsString_v2(individual.item)} with assigned probability to be drawn = $assignedProb" }.mkString("  ,  ")} ]")

      val selector = new RouletteWheel[EvaluatedTemplateData](evaluatedTemplateDataWithAssignedProbs)
      debug(s"Selector RouletteWheel is created")
      val selected = selector.sample(1).map(_._1)

      debug(s"Selected: ${selected.head.idShort}")

      (selectedAcc ++ selected, rest.diff(selected))
    }
    }
  }

  def parentSelectionByShareWithLocalCompetitions(selectionShare: Double, individuals: Seq[EvaluatedTemplateData]): Seq[EvaluatedTemplateData] = {
    require(individuals.forall(_.neighbours.nonEmpty) , "We should not call local competition version without having neighbours")
    val numberOfParents = ratioToSIze(selectionShare, individuals)

    val res = selectionBySizeWithLocalCompetitions(numberOfParents, individuals)
    info("parentSelectionByShareWithLocalCompetitions is finished")
    res
  }

  def parentSelectionBySize(numberOfParentsToSelect: Int, individuals: Seq[EvaluatedTemplateData]): Seq[EvaluatedTemplateData] = {
    val orderedByFitness = individuals.sortWith((l,r) => l.fitness.orderTo(r.fitness))

    val numberOfCompetitors = individuals.length
    val linearRankingProbabilityStrategy = new LinearRankingProbabilityStrategy(numberOfCompetitors, parameter_S = 1.5) // TODO replace with RankBasedSelectionProbabilityAssigner and RouletteWheel

    val ranked: Seq[EvaluatedTemplateData] = orderedByFitness
      .zipWithIndex
      .map { case (ind, rank) =>
        ind.withRank(rank).withProbability(linearRankingProbabilityStrategy.computeProbabilityFor(rank))
      }

    val rankedWithCumulativeProbs = ranked.drop(1).scanLeft(ranked.head){ case (acc, indd2) => indd2.copy(probability = indd2.probability + acc.probability)}
    debug(rankedWithCumulativeProbs.map(r => (r.rank, r.probability)).mkString("\n"))

    var currentParentIndex = 0
    val selectedParents = new Array[EvaluatedTemplateData](numberOfParentsToSelect)
    while (currentParentIndex < numberOfParentsToSelect) {
      val r = Random.nextDouble()
      val rouletteWheel = rankedWithCumulativeProbs.dropWhile(individualData => individualData.probability < r)
      val selected = rouletteWheel.headOption.getOrElse(rankedWithCumulativeProbs.last)
      debug(s"Selected for r = $r : $selected")
      selectedParents(currentParentIndex) = selected
      currentParentIndex += 1
    }
    selectedParents // TODO choose optimal data structure. Implicit conversion here.
  }

  private def ratioToSIze(selectionShare: Double, individuals: Seq[EvaluatedTemplateData]) = {
    require(selectionShare < 1 && selectionShare > 0, "Selection share parameter shoud be in range (0, 1)")
    val numberOfCompetitors = individuals.length
    val numberOfParents = (numberOfCompetitors * selectionShare).toInt
    numberOfParents
  }
}

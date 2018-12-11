package com.automl.evolution.selection

import com.typesafe.scalalogging.LazyLogging

import scala.util.Random

/**
  *  Cumulative Probability Selector
  *  ( see 5.2 Parent Selection page 83 of Introduction to Evolutionary computing for details)
  * @param items
  * @tparam T
  */

class RouletteWheel[T](items: List[( T, Double)]) extends Selector[T] with LazyLogging {

  val sumOfGivenProbabilities: Double = items.map(_._2).sum
  if(sumOfGivenProbabilities > 1) //TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! fix this to require
    logger.warn( s"Roulette selector requires that all probabilities add up to one but it was: $sumOfGivenProbabilities")
//  require(sumOfGivenProbabilities == 1, s"Roulette selector requires that all probabilities add up to one but it was: $sumOfGivenProbabilities")
  private val sortedAscendingItems: List[(T, Double)] = items.sortWith(_._2 < _._2)
  private lazy val rankedWithCumulativeProbs = sortedAscendingItems.drop(1).scanLeft(sortedAscendingItems.head){
    case (acc, (item, probability)) => (item, acc._2 + probability)
  }

  def getNext: (T, Double) = {
    val r = Random.nextDouble()
    val rouletteWheel = rankedWithCumulativeProbs.dropWhile{case (item, cumProbability) => cumProbability < r}
    val selected = rouletteWheel.headOption.getOrElse(throw new IllegalStateException("There is no elements in the CumulativeProbabilitySelector"))
    selected
  }
}

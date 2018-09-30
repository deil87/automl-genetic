package com.automl.evolution.selection

import scala.util.Random

trait Distributed {
  def probability: Double
}
class CumulativeProbabilitySelector[T](items: List[( T, Double)]) {

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

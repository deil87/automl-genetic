package com.automl.helper

class LinearRankingProbabilityStrategy(populationSize: Long, parameter_S: Double ) {

  def computeProbabilityFor(rank: Long): Double = {
    ((2 - parameter_S) / populationSize) + 2 * rank * (parameter_S -1) / (populationSize * (populationSize - 1))
  }

}

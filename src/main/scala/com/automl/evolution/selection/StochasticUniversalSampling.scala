package com.automl.evolution.selection

import scala.util.Random

trait Distributed {
  def probability: Double
}

/**
  *  Stochastic Universal Sampling Selector
  *  ( see 5.2 Parent Selection page 84 of Introduction to Evolutionary computing for details)
  * @param items
  * @tparam T
  */

class StochasticUniversalSampling[T](items: List[( T, Double)]) extends Selector[T] {

  override def sample(sizeOfSample: Int): Stream[(T, Double)] = ???

  def getNext: (T, Double) = { ??? //TODO

  }
}

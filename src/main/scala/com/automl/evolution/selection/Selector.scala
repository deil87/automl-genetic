package com.automl.evolution.selection

trait Selector[T] {
  def getNext: (T, Double)
  def sample(sizeOfSample: Int): Stream[(T, Double)]
}

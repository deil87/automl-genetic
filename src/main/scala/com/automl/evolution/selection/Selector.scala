package com.automl.evolution.selection

trait Selector[T] {
  def getNext: (T, Double)
}

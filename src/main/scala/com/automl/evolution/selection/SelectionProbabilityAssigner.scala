package com.automl.evolution.selection

trait SelectionProbabilityAssigner[T] {

  def assign(items: List[T]): List[(T, Double)]
}

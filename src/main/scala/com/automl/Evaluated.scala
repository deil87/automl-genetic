package com.automl

trait Evaluated[T <: Evaluated[T]] {
  type ItemType
  type FitnessType

  def item: ItemType
  def result: FitnessType

  def compare(other: T): Boolean

}
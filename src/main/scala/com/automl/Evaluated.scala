package com.automl

trait Evaluated[T <: Evaluated[T]] {
  type ItemType
  type FitnessType
  type ParamsType

  def item: ItemType
  def result: FitnessType
  def params: ParamsType

  def compare(other: T): Boolean

}
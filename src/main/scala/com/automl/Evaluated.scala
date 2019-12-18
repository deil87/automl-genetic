package com.automl

trait Evaluated[T <: Evaluated[T]] extends Ordered[T] with EvaluationContext {
  type ItemType
  type FitnessType
  type ParamsType

  def item: ItemType
  def result: FitnessType
  def params: Option[ParamsType]

  def idShort: String

}
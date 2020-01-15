package com.automl

import com.automl.evolution.evaluation.EvaluationContextInfo

import scala.concurrent.{Future, Promise}

trait Evaluated[T <: Evaluated[T]] extends Ordered[T] with EvaluationRules {
  type ItemType
  type FitnessType
  type ParamsType

  def item: ItemType
  def result: FitnessType

  protected val evaluationContextInfoPromise: Promise[EvaluationContextInfo] = Promise[EvaluationContextInfo]()
  def setEvaluationContextInfo(eci: EvaluationContextInfo): Unit = {
    evaluationContextInfoPromise.success(eci)
  }
  def evaluationContextInfo: Future[EvaluationContextInfo] = evaluationContextInfoPromise.future

  def params: Option[ParamsType]

  def idShort: String

}
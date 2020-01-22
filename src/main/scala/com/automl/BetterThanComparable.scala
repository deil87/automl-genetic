package com.automl

import com.automl.evolution.evaluation.EvaluationContextInfo

import scala.concurrent.{Future, Promise}

trait BetterThanComparable[FitnessType] {

  def betterThan(that:FitnessType): Boolean

  def betterThanOrEqual(that:FitnessType): Int

}
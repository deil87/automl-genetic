package com.automl.template

import com.automl.helper.FitnessResult
import org.apache.spark.sql.DataFrame


sealed trait EvaluationMagnet {
  type Result
  def apply(): Result
}

object EvaluationMagnet {
  implicit def fromSimpleModel(tuple : ( DataFrame, DataFrame)) =
    new EvaluationMagnet {
      type Result = FitnessResult

      def apply(): Result = FitnessResult(42, ???)

    }

  implicit def fromEnsemblingModel[A <: TemplateMember](tuple: ( DataFrame, DataFrame, Seq[TemplateTree[A]])) =
    new EvaluationMagnet {
      type Result = FitnessResult

      def apply(): Result = {
        val (trainDF, testDF, subMembers) = tuple
        subMembers.map(_.evaluateFitness(trainDF, testDF))
        FitnessResult(42, ???)
      }

    }
}
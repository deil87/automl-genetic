package com.automl.template

import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import org.apache.spark.sql.DataFrame


sealed trait EvaluationMagnet {
  type Result
  def apply(): Result
}

object EvaluationMagnet {
  implicit def fromSimpleModel(tuple : ( DataFrame, DataFrame)) =
    new EvaluationMagnet {
      type Result = FitnessResult

      def apply(): Result = FitnessResult(???, ???, ???)

    }

  implicit def fromEnsemblingModel[A <: TemplateMember](tuple: ( DataFrame, DataFrame, Seq[TemplateTree[A]], ProblemType)) =
    new EvaluationMagnet {
      type Result = FitnessResult

      def apply(): Result = {
        val (trainDF, testDF, subMembers, problemType) = tuple
        subMembers.map(_.evaluateFitness(trainDF, testDF, problemType))
        FitnessResult(???, ???, ???)
      }

    }
}
package com.automl.template.simple.perceptron

import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.EvaluationMagnet
import com.automl.template.simple.{LinearModelMember, SimpleModelMember}
import org.apache.spark.sql._

case class LinearPerceptron() extends LinearModelMember {
  override def name: String = "LinearPerceptron " + super.name


  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case MultiClassClassificationProblem => ???
    case BinaryClassificationProblem => ???
    case RegressionProblem => ???
  }

  lazy val predictor = ???

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???


  override def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType): FitnessResult = ???

}

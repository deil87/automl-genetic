package com.automl.problemtype

sealed trait ProblemType

object ProblemType{
  case object RegressionProblem extends ProblemType
  case object ClassificationProblem extends ProblemType
}

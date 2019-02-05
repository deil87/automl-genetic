package com.automl.problemtype

sealed trait ProblemType {
  def isClassification: Boolean
}

object ProblemType{
  case object RegressionProblem extends ProblemType {
    override def isClassification: Boolean = false
  }
  case object BinaryClassificationProblem extends ProblemType {
    override def isClassification: Boolean = true
  }
  case object MultiClassClassificationProblem extends ProblemType {
    override def isClassification: Boolean = true
  }
}

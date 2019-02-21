package com.automl.template.simple.perceptron

import com.automl.PaddedLogging
import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.EvaluationMagnet
import com.automl.template.simple.{LinearModelMember, SimpleModelMember}
import org.apache.spark.sql._

case class LinearPerceptron()(implicit val logPaddingSize: Int = 0) extends LinearModelMember with PaddedLogging{
  override def name: String = "LinearPerceptron " + super.name


  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case MultiClassClassificationProblem => true
    case BinaryClassificationProblem => true
    case RegressionProblem => ???
  }

  lazy val predictor = ???

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???


  override def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType): FitnessResult = ???

}

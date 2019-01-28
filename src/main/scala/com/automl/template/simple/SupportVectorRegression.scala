package com.automl.template.simple

import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.EvaluationMagnet
import org.apache.spark.sql._

/*
* Support Vector Machines were developed for binary classification problems,
* although extensions to the technique have been made to support multi-class classification and regression problems.
 * The adaptation of SVM for regression is called Support Vector Regression or SVR for short.
* */
case class SupportVectorRegression() extends LinearModelMember {
  override def name: String = "SupportVectorRegression " + super.name


  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case MultiClassClassificationProblem => true
    case BinaryClassificationProblem => true
    case RegressionProblem => true
  }

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

  override def fitnessError(trainDF: DataFrame, testDF: DataFrame): FitnessResult = ???

}

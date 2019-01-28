package com.automl.template.simple

import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.EvaluationMagnet
import com.automl.teststrategy.{TestStrategy, TrainingTestSplitStrategy}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql._

class LogisticRegressionModel() extends LinearModelMember with LazyLogging{

  override def name: String = "LogisticRegressionModel " + super.name

  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case BinaryClassificationProblem => true
    case MultiClassClassificationProblem => true
    case RegressionProblem => false
  }

  override def testStrategy: TestStrategy = new TrainingTestSplitStrategy()


  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

  //We need here a task/problemType or something like ModelSearchContext()
  override def fitnessError(trainDF: DataFrame, testDF: DataFrame): FitnessResult = {

    logger.debug(s"Evaluating $name ...")
    val model = new LogisticRegression()
    // TODO https://spark.apache.org/docs/2.3.1/mllib-linear-methods.html
//    val problemType: ProblemType = _
//    problemType match {
//      case BinaryClassificationProblem =>
//      case MultiClassClassificationProblem =>
//    }
//    logger.info(s"Area under ROC = $auROC")
//    import testDF.sparkSession.implicits._
//    val predictionsAsDF = scoreAndLabels.toDF("score", "prediction") //TODO maybe we need to join scores and labels with original data here

    FitnessResult(???, ???)
  }
}

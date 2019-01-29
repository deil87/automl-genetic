package com.automl.template.simple

import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.EvaluationMagnet
import com.automl.teststrategy.{TestStrategy, TrainingTestSplitStrategy}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._

class SVMModel() extends LinearModelMember with LazyLogging{

  override def name: String = "SVMModel " + super.name

  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case BinaryClassificationProblem => true
    case MultiClassClassificationProblem => false
    case RegressionProblem => false
  }

  override def testStrategy: TestStrategy = new TrainingTestSplitStrategy()


  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???


  override def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType): FitnessResult = {

    import utils.SparkMLUtils._
    require(problemType == BinaryClassificationProblem) //TODO maybe this check is unnecessary

    logger.debug(s"Evaluating $name ...")
    val numIterations = 100
    val trainingLabeledPoints = trainDF.toMLLibLabelPoint.rdd
    val model = SVMWithSGD.train(trainingLabeledPoints, numIterations, 1.0, 0.01, 1.0) // Looks like there is no version in spark.ml

    model.clearThreshold()

    val scoreAndLabels: RDD[(Double, Double)] = testDF.toMLLibLabelPoint.rdd.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }


    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    logger.info(s"Area under ROC = $auROC")
    import testDF.sparkSession.implicits._
    val predictionsAsDF = scoreAndLabels.toDF("score", "prediction") //TODO maybe we need to join scores and labels with original data here

    FitnessResult(Map("auc" -> auROC), problemType, predictionsAsDF)
  }
}

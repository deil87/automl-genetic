package com.automl.template.simple

import com.automl.PaddedLogging
import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.EvaluationMagnet
import com.automl.teststrategy.{TestStrategy, TrainingTestSplitStrategy}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.classification.{LinearSVC, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._

case class SVMModel()(implicit val logPaddingSize: Int = 0) extends LinearModelMember with PaddedLogging{

  override def name: String = "SVMModel " + super.name

  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case BinaryClassificationProblem => true
    case MultiClassClassificationProblem => true
    case RegressionProblem => false
  }

  override def testStrategy: TestStrategy = new TrainingTestSplitStrategy()


  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???


  override def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType, hyperParametersField: Option[HyperParametersField]): FitnessResult = {
    logger.debug(s"Evaluating $name ...")
    import utils.SparkMLUtils._
    problemType match {
      case BinaryClassificationProblem =>

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

      case MultiClassClassificationProblem =>

        trainDF.cache()
        testDF.cache()

        trainDF.showAllAndContinue
        val lsvc = new LinearSVC()
          .setMaxIter(10)
          .setRegParam(0.1)
          .setLabelCol("indexedLabel")
          .setStandardization(true)
          .setFitIntercept(true)
          .setTol(1e-3) // TODO estimate?
          //.setThreshold()

        val ovr = new OneVsRest()
          .setClassifier(lsvc)
          .setFeaturesCol("features")
          .setLabelCol("indexedLabel")

        val ovrModel = ovr.fit(trainDF)

        val predictions = ovrModel.transform(testDF).cache()

        MulticlassMetricsHelper.showStatistics(predictions)

        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("indexedLabel")
          .setMetricName("f1")

        val f1 = evaluator.evaluate(predictions)
        logger.info(s"Finished. $name : F1 = " + f1)

        FitnessResult(Map("f1" -> f1), problemType, predictions)
      case RegressionProblem => throw new UnsupportedOperationException("Regression is not supported by SVMModel")
    }
  }
}

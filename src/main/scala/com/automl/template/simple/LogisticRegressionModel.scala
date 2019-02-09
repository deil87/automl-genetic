package com.automl.template.simple

import com.automl.evolution.dimension.hparameter.{LRRegParam, LogisticRegressionHPGroup}
import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.EvaluationMagnet
import com.automl.teststrategy.{TestStrategy, TrainingTestSplitStrategy}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.sql._

case class LogisticRegressionModel(hpGroup: LogisticRegressionHPGroup = LogisticRegressionHPGroup.default) extends LinearModelMember with LazyLogging{

  override def name: String = "LogisticRegressionModel " + super.name

  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case BinaryClassificationProblem => true
    case MultiClassClassificationProblem => true
    case RegressionProblem => false
  }

  import utils.SparkMLUtils._

  override def testStrategy: TestStrategy = new TrainingTestSplitStrategy()


  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

  //We need here a task/problemType or something like ModelSearchContext()
  override def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType): FitnessResult = {

  logger.debug(s"Evaluating $name ...")
    problemType match {
      case BinaryClassificationProblem =>
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
        FitnessResult(???, ???, ???)
      case MultiClassClassificationProblem =>
        val scaler = new StandardScaler()
          .setInputCol("features")
          .setOutputCol("scaledFeatures")
          .setWithStd(true)
          .setWithMean(false)

        val lrEstimator = new LogisticRegression()
          .setLabelCol("indexedLabel")
          .setMaxIter(20)
          .setElasticNetParam(0.8)

        val lrWithHP = hpGroup.hpParameters.foldLeft(lrEstimator)((res, next) => next match {
          case p@LRRegParam() =>
            logger.debug(s"LogisticRegression lambda hyper-parameter was set to ${p.currentValue}")
            res.setRegParam(p.currentValue)
        })

        val preparedTrainingDF = trainDF
          .applyTransformation(scaler)
          .drop("features")
          .withColumnRenamed("scaledFeatures", "features")
          .cache()

        val lrModel = lrWithHP.fit(preparedTrainingDF)
        val prediction = lrModel.transform(testDF)

        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("indexedLabel")
          .setMetricName("f1")

        val f1 = evaluator.evaluate(prediction)
        logger.info(s"Finished. $name : F1 = " + f1)

        FitnessResult(Map("f1" -> f1), problemType, prediction)
    }

  }
}

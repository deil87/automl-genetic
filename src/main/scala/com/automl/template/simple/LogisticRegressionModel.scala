package com.automl.template.simple

import com.automl.{ConfigProvider, PaddedLogging}
import com.automl.evolution.dimension.hparameter._
import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.EvaluationMagnet
import com.automl.teststrategy.{TestStrategy, TrainingTestSplitStrategy}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql._

case class LogisticRegressionModel(hpGroup: Option[LogisticRegressionHPGroup] = None)(implicit val logPaddingSize: Int = 0)
  extends LinearModelMember
    with ClassificationMetricsHelper
    with PaddedLogging{

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
  override def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType, hyperParametersField: Option[HyperParametersField] = None): FitnessResult = {

  debug(s"Evaluating $name ...")
    problemType match {
      case BinaryClassificationProblem =>
        val model = new LogisticRegression()
        // TODO https://spark.apache.org/docs/2.3.1/mllib-linear-methods.html
        //    val problemType: ProblemType = _
        //    problemType match {
        //      case BinaryClassificationProblem =>
        //      case MultiClassClassificationProblem =>
        //    }
        //    info(s"Area under ROC = $auROC")
        //    import testDF.sparkSession.implicits._
        //    val predictionsAsDF = scoreAndLabels.toDF("score", "prediction") //TODO maybe we need to join scores and labels with original data here
        FitnessResult(???, ???, ???)
      case MultiClassClassificationProblem =>
        val config = ConfigProvider.config.getConfig("evolution")

        val validationStrategy = config.getString("templateDimension.validationStrategy")

        val scaler = new StandardScaler()
          .setInputCol("features")
          .setOutputCol("scaledFeatures")
          .setWithStd(true)
          .setWithMean(false)

        val lrEstimator = new LogisticRegression()
          .setLabelCol("indexedLabel")
          .setMaxIter(20)

        val activeHPGroup = getActiveHPGroup(config, hpGroup, hyperParametersField)


        val preparedTrainingDF = trainDF
          .applyTransformation(scaler)
          .drop("features")
          .withColumnRenamed("scaledFeatures", "features")
          .cache()

        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("indexedLabel")
          .setMetricName("f1")

        if(validationStrategy == "cv") {
          val paramGrid = new ParamGridBuilder()
          val configuredParamGrid = activeHPGroup.hpParameters.foldLeft(paramGrid)((res, next) => next match {
            case p@RegParamLR(_) =>
              debug(s"LogisticRegression's regParam hyper-parameter was set to ${p.currentValue}")
              res.addGrid(lrEstimator.regParam, Array(p.currentValue))
            case p@ElasticNet(_) =>
              debug(s"LogisticRegression's elasticNetParam hyper-parameter was set to ${p.currentValue}")
              res.addGrid(lrEstimator.elasticNetParam, Array(p.currentValue))
          }).build()
          val cv = new CrossValidator()
            .setEstimator(lrEstimator)
            .setEvaluator(evaluator)
            .setEstimatorParamMaps(configuredParamGrid)
            .setNumFolds(3)
            .setParallelism(2) // TODO 2 or ??

          val modelCV = cv.fit(trainDF)
          val f1CV = modelCV.avgMetrics(0) // <- this is averaged metric whereas `evaluator.setMetricName("f1").evaluate(predictions)` will return metric computed only on test data
          val predictions = modelCV.transform(testDF)
          //Unused
          //          val metrics = new MulticlassMetrics(predictions.select("prediction", "indexedLabel").rdd.map(r => (r.getDouble(0), r.getDouble(1))))
          printConfusionMatrix(predictions, testDF)

          FitnessResult(Map("f1" -> f1CV, "accuracy" -> -1), problemType, predictions)
        } else {
          val lrWithHP = activeHPGroup.hpParameters.foldLeft(lrEstimator)((res, next) => next match {
            case p@RegParamLR(_) =>
              debug(s"LogisticRegression lambda hyper-parameter was set to ${p.currentValue}")
              res.setRegParam(p.currentValue)
            case p@ElasticNet(_) =>
              debug(s"LogisticRegression elastic_net hyper-parameter was set to ${p.currentValue}")
              res.setElasticNetParam(p.currentValue)
          })
          val lrModel = lrWithHP.fit(preparedTrainingDF)
          val prediction = lrModel.transform(testDF).cache()


          val f1 = evaluator.evaluate(prediction)
          info(s"Finished. $name : F1 = " + f1)

          FitnessResult(Map("f1" -> f1), problemType, prediction)
        }


      case RegressionProblem => throw new UnsupportedOperationException("Regression is not supported by LogisticRegressionModel")
    }

  }
}

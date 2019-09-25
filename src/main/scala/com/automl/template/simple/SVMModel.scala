package com.automl.template.simple

import com.automl.evolution.dimension.hparameter._
import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.EvaluationMagnet
import com.automl.teststrategy.{TestStrategy, TrainingTestSplitStrategy}
import com.automl.{ConfigProvider, PaddedLogging}
import org.apache.spark.ml.classification.{LinearSVC, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._

import scala.util.Random

case class SVMModel(hpGroup: Option[SVMHPGroup] = None, seed: Long = Random.nextLong())(implicit val logPaddingSize: Int = 0)
  extends LinearModelMember
    with ClassificationMetricsHelper
    with PaddedLogging{

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

        throw new UnsupportedOperationException("SVMModel should be tested for BinaryClassificationProblem or we can merge it with MultiClassClassificationProblem")
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

//        trainDF.showAllAndContinue
        val lsvc = new LinearSVC()
//          .setMaxIter(10)
//          .setRegParam(0.1)
          .setLabelCol("indexedLabel")
          .setStandardization(true)
          .setFitIntercept(true)
          .setTol(1e-3) // TODO estimate?
          //.setThreshold()

        val numIterations = 100
        val trainingLabeledPoints = trainDF.toMLLibLabelPoint.rdd
        val model = SVMWithSGD.train(trainingLabeledPoints, numIterations, 1.0, 0.01, 1.0)

        val ovr = new OneVsRest()
          .setClassifier(lsvc)
          .setFeaturesCol("features")
          .setLabelCol("indexedLabel")

        val config = ConfigProvider.config.getConfig("evolution")
        //        val performGridSearch = config.getBoolean("hpGridSearch")
        val validationStrategy = config.getString("templateDimension.validationStrategy")
        //        val hpCoevolutionIsEnabled = config.getBoolean("hyperParameterDimension.enabled")

        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("indexedLabel")
          .setPredictionCol("prediction")
          .setMetricName("f1")

        val activeHPGroup = getActiveHPGroup(config, hpGroup, hyperParametersField)

        // We can't train CV on `train+test` data and then predict on itself -> overfitted resuls.
        // We need at least `test` split  to get predictions which could be used to find phenotypic similarity.
        // But we can use CV averaged estimate instead of estimate on test split
        if(validationStrategy == "cv") {
          val paramGrid = new ParamGridBuilder()
          val configuredParamGrid = activeHPGroup.hpParameters.foldLeft(paramGrid)((res, next) => next match {
            case p@MaxIterSVM(_) =>
              debug(s"SVM's max_depth_rf hyper-parameter was set to ${p.currentValue}")
              res.addGrid(lsvc.maxIter, Array(p.currentValue.toInt))
            case p@RegParamSVM(_) =>
              debug(s"SVM's max_depth_rf hyper-parameter was set to ${p.currentValue}")
              res.addGrid(lsvc.regParam, Array(p.currentValue))
          }).build()
          val cv = new CrossValidator()
            .setEstimator(ovr)
            .setEvaluator(evaluator)
            .setEstimatorParamMaps(configuredParamGrid)
            .setNumFolds(3) // Config: nfolds for CV
            .setParallelism(2) // TODO 2 or ??
            .setSeed(seed)

          val modelCV = cv.fit(trainDF) // TODO maybe we need to make testDF to be optional and used trainingDF as CV
          val f1CV = modelCV.avgMetrics(0) // <- this is averaged metric whereas `evaluator.setMetricName("f1").evaluate(predictions)` will return metric computed only on test data
          val predictions = modelCV.transform(testDF)

          MulticlassMetricsHelper.showStatistics(predictions)
          printConfusionMatrix(predictions, testDF)

          //Unused
          val f1 = evaluator.setMetricName("f1").evaluate(predictions)

          FitnessResult(Map("f1" -> f1CV, "accuracy" -> -1), problemType, predictions)
        } else {
          throw new IllegalStateException("Only CV validation strategy for SVM is supported.")
        }

      case RegressionProblem =>
        throw new UnsupportedOperationException("Regression is not supported by SVMModel")
    }
  }
}

package com.automl.template.simple

import com.automl.evolution.dimension.hparameter._
import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.EvaluationMagnet
import com.automl.{ConfigProvider, LogLossCustom, PaddedLogging}
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql._

case class XGBoostEstimator(hpGroup: DecisionTreeHPGroup = DecisionTreeHPGroup())(implicit val logPaddingSize: Int = 0) extends SimpleModelMember with PaddedLogging{
  override def name: String = "XGBoost-Spark " + super.name

  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case BinaryClassificationProblem => false
    case MultiClassClassificationProblem => true
    case RegressionProblem => false
  }

  var hpGroupInternal: HyperParametersGroup[_ <: MutableHParameter[Double, _]] = hpGroup

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = null

  override def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType, hpFieldFromCoevolution: Option[HyperParametersField]): FitnessResult = {

    problemType match {
      case RegressionProblem => throw new UnsupportedOperationException()

      case MultiClassClassificationProblem =>
        val config = ConfigProvider.config.getConfig("evolution")

//        val validationStrategy = config.getString("templateDimension.validationStrategy")

        val xgbParam = Map("eta" -> 0.1f,
          "missing" -> -999,
//          "objective" -> "multi:softmax",
          "objective" -> "multi:softprob",
          "num_class" -> 3,
          "num_round" -> 10,
          "num_workers" -> 2)

        val xgbClassifier = new XGBoostClassifier(xgbParam).
          setFeaturesCol("features").
          setLabelCol("indexedLabel")

        import org.apache.spark.ml.tuning._
        import org.apache.spark.ml.PipelineModel
        import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel

        val paramGrid = new ParamGridBuilder()
          .addGrid(xgbClassifier.maxDepth, Array(3, 8))
          .addGrid(xgbClassifier.eta, Array(0.2, 0.6))
          .build()



        val activeHPGroup: HyperParametersGroup[_] = getRelevantHPGroupFromActiveHPField(config, hpFieldFromCoevolution).getOrElse(hpGroupInternal)

        val preparedTrainingDF = trainDF
//          .drop("features")
//          .withColumnRenamed("scaledFeatures", "features")
          .cache()

        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("indexedLabel")
          .setMetricName("f1")

        val cv = new CrossValidator()
          .setEstimator(xgbClassifier)
          .setEvaluator(evaluator)
          .setEstimatorParamMaps(paramGrid)
          .setNumFolds(3)

        val xgbClassifierWithHP = activeHPGroup.hpParameters.foldLeft(xgbClassifier)((classifier, next) => next match {
          case p@RegParamLR(_) =>
//            debug(s"LogisticRegression's regParam hyper-parameter was set to ${p.currentValue}")
//            classifier.set(p.currentValue)
            throw new UnsupportedOperationException()
          case p@ElasticNet(_) =>
            throw new UnsupportedOperationException()
//            debug(s"LogisticRegression elastic_net hyper-parameter was set to ${p.currentValue}")
//            classifier.setElasticNetParam(p.currentValue)
          case _ => classifier
        })
//        val xGBoostClassificationModel = xgbClassifierWithHP.fit(preparedTrainingDF)
        val xGBoostClassificationModel = cv.fit(preparedTrainingDF)
        val predictions = xGBoostClassificationModel.transform(testDF).cache()


        val f1 = evaluator.evaluate(predictions)
        val logLoss = 42.0 // LogLossCustom.compute(predictions)
        info(s"Finished. $name : F1 = $f1, logloss = $logLoss")

        FitnessResult(Map("f1" -> f1, "logloss" -> logLoss), problemType, predictions)

      case BinaryClassificationProblem =>
        throw new UnsupportedOperationException()

    }
  }

}

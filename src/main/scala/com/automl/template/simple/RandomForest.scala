package com.automl.template.simple

import com.automl.{ConfigProvider, LogLossCustom, PaddedLogging}
import com.automl.evolution.dimension.hparameter.{DecisionTreeHPGroup, HyperParametersField, MaxDepth, MaxDepthRF}
import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.spark.SparkSessionProvider
import com.automl.template.{EvaluationMagnet, ModelKey}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql._

import scala.util.Random

case class RandomForest(hpGroup: Option[DecisionTreeHPGroup] = None, seed: Long = Random.nextLong())(implicit val logPaddingSize: Int = 0)
  extends SimpleModelMember
    with ClassificationMetricsHelper
    with SparkSessionProvider
    with PaddedLogging{
  override def name: String = "Random forest " + super.name


  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case BinaryClassificationProblem => true
    case MultiClassClassificationProblem => true
    case RegressionProblem => true
  }

  override def modelKey: ModelKey = ModelKey("RandomForest")

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???


  override def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType, hyperParametersField: Option[HyperParametersField]): FitnessResult = {

    problemType match {
      case RegressionProblem =>
        //TODO WARNING!!!!!!!!!!!!!!! this is old version.Recheck.
        //We can iterate over HLIST and use
        // new RandomForestRegressor().set()
        val rfr =  new RandomForestRegressor()
          .setLabelCol("label")
          .setFeaturesCol("indexedFeatures")

        val featureIndexer = new VectorIndexer()
          .setInputCol("features")
          .setOutputCol("indexedFeatures")
          .setMaxCategories(4)
          .fit(trainDF.union(testDF))

        val pipeline = new Pipeline()
          .setStages(Array(featureIndexer, rfr))

        val model = pipeline.fit(trainDF)
        val predictions = model.transform(testDF)
          .drop("features")
          .withColumnRenamed("indexedFeatures", "features")

        val evaluator = new RegressionEvaluator()
        val rmse: Double = evaluator.evaluate(predictions)

        logger.info(s"$name : RMSE = " + rmse)
        FitnessResult(Map("rmse" -> rmse), problemType, predictions)


      case MultiClassClassificationProblem | BinaryClassificationProblem =>

        val rf = new RandomForestClassifier()
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
            case p@MaxDepthRF(_) =>
              debug(s"Random Forest's max_depth_rf hyper-parameter was set to ${p.currentValue}")
              res.addGrid(rf.maxDepth, Array(p.currentValue.toInt))
          }).build()
          val cv = new CrossValidator()
            .setEstimator(rf)
            .setEvaluator(evaluator)
            .setEstimatorParamMaps(configuredParamGrid)
            .setNumFolds(3) // Config: nfolds for CV
            .setParallelism(2) // TODO 2 or ??
            .setSeed(seed)

          val modelCV = cv.fit(trainDF) // TODO maybe we need to make testDF to be optional and used trainingDF as CV
          val f1CV = modelCV.avgMetrics(0) // <- this is averaged metric whereas `evaluator.setMetricName("f1").evaluate(predictions)` will return metric computed only on test data
          val predictions = modelCV.transform(testDF)

          printConfusionMatrix(predictions, testDF)

          val logLoss = LogLossCustom.compute(predictions)

          //Unused
          val f1 = evaluator.setMetricName("f1").evaluate(predictions)

          FitnessResult(Map("f1" -> f1CV, "accuracy" -> -1, "logloss" -> logLoss), problemType, predictions)
        } else {
          throw new IllegalStateException("Only CV strategy for RandomForest is supported.")
        }
    }
  }

}

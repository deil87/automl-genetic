package com.automl.template.ensemble.stacking

import com.automl.ConfigProvider
import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.classifier.ensemble.stacking.SparkGenericStacking
import com.automl.dataset.Datasets
import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.problemtype.ProblemType
import com.automl.spark.SparkSessionProvider
import com.automl.template._
import com.automl.template.simple._
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.regression.{GBTRegressor, LinearRegression}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.storage.StorageLevel
import org.scalatest.{FunSuite, Matchers}
import com.test.TestBase
import utils.{BenchmarkHelper, SparkMLUtils}

import scala.util.Random


class GenericStackingRegressionSuite extends FunSuite with SparkSessionProvider with TestBase{

  import utils.SparkMLUtils._
  import ss.implicits._

  test("We can train on this dataset with base members") {

    val prepairedAirlineDF = Datasets.getAirlinesDataFrameRegression(1234)

    val Array(trainingSplit, testSplit) = prepairedAirlineDF.randomSplit(Array(0.8, 0.2))

    trainingSplit.persist(StorageLevel.MEMORY_ONLY)
    testSplit.persist(StorageLevel.MEMORY_ONLY)

    val linearRegression = new LinearRegression()

    noException shouldBe thrownBy(  linearRegression.fit(trainingSplit))

  }

  ignore("Generic stacking member should calculate fitness over complex tree of members and with better performance than each particular member") {

    val prepairedAirlineDF = Datasets.getAirlinesDataFrameRegression(1234)

    val Array(trainingSplit, testSplit) = prepairedAirlineDF.randomSplit(Array(0.8, 0.2))

    val models = Seq(
      LeafTemplate(new LinearRegressionModel()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(RandomForest()),
//      LeafTemplate(Bayesian()), // TODO why Bayesian model is so bad?
      NodeTemplate(SparkGenericBagging(), Seq(
        LeafTemplate(new LinearRegressionModel()),
        LeafTemplate(GradientBoosting()),
        LeafTemplate(DecisionTree())
      ))
    )

    val genericStacking = GenericStacking(unusedMetaLearner = new GBTRegressor())

    val problemType = ProblemType.RegressionProblem

    val seed = new Random().nextLong()

    val fitnessResult = genericStacking.ensemblingFitnessError(trainingSplit, testSplit, models, problemType, hyperParamsField = Some(HyperParametersField.default), seed = seed)

    val rmseFromLR = LinearRegressionModel().fitnessError(trainingSplit, testSplit).getCorrespondingMetric
    println(s"RMSE computed for Linear regression model $rmseFromLR")

    fitnessResult.getCorrespondingMetric should be <= rmseFromLR

    val rmseFromGB = GradientBoosting().fitnessError(trainingSplit, testSplit).getCorrespondingMetric
    println(s"RMSE computed for GradientBoosting model $rmseFromGB")

    fitnessResult.getCorrespondingMetric should be <= rmseFromGB

  }

  test("KFold util method should not return 0 folds") {
    val prepairedAirlineDF = Datasets.getAirlinesDataFrameRegression(1234).limit(15)
    val seedForFoldSplitting = 3456
    val splits = MLUtils.kFold(prepairedAirlineDF.rdd, 3, seedForFoldSplitting)
    splits.forall{ case (train, test) => train.count() > 0 && test.count() > 0} shouldBe true

    val trainLevel2 = splits.apply(0)._1

    val splitsLevel2 = MLUtils.kFold(trainLevel2, 3, seedForFoldSplitting)
    val splitCounts = splitsLevel2.map(res => (res._1.count() , res._2.count()))

//    splitsLevel2.apply(0)._2.count() shouldBe(0)
  }

  test("Generic stacking member should calculate fitness over tree of height 3 and with better performance than each particular member") {

    val metric = "logloss"
    ConfigProvider.clearOverride.addOverride(
      s"""
         |evolution {
         |  hpGridSearch = false
         |  hyperParameterDimension {
         |    enabled = false
         |  }
         |  templateDimension {
         |   validationStrategy = "tt"
         |  }
         |  evaluation {
         |    multiclass.metric = "$metric"
         |  }
         |}
      """)

    val iris = Datasets.getIrisDataFrame(1234)

    val Array(trainingSplit, testSplit) = iris.randomSplit(Array(0.8, 0.2))

    val models = Seq(
      LeafTemplate(DecisionTree()),
      LeafTemplate(LogisticRegressionModel()),
      LeafTemplate(LogisticRegressionModel())/*,
      NodeTemplate(GenericStacking(), Seq(
        NodeTemplate(SparkGenericBagging(), Seq(
          LeafTemplate(LogisticRegressionModel()),
          LeafTemplate(DecisionTree())
        )),
        NodeTemplate(SparkGenericBagging(), Seq(
          LeafTemplate(LogisticRegressionModel()),
          LeafTemplate(DecisionTree())
        ))
      ))*/
    )

    BenchmarkHelper.time("GenericStackingSuite.scala ") {

      val genericStacking = GenericStacking(unusedMetaLearner = new GBTRegressor())

      val problemType = ProblemType.MultiClassClassificationProblem

      val seed = new Random().nextLong()

      val gsMetricValue = genericStacking.ensemblingFitnessError(trainingSplit, testSplit, models, problemType, hyperParamsField = None, seed = seed)

      gsMetricValue.dfWithPredictions.show(5, false)

      val lrMetricValue = LogisticRegressionModel().fitnessError(trainingSplit, testSplit, problemType, hpFieldFromCoevolution = None).getCorrespondingMetric

      logger.debug(s"Stacking logloss: ${gsMetricValue.getCorrespondingMetric}, LogisticRegression logloss: $lrMetricValue")
      gsMetricValue.getCorrespondingMetric should be <= lrMetricValue

    }(0)
  }


  ignore("Should combine models for stacking") {
    val observations = ss.sparkContext.parallelize(
      Seq(
        (1, 1, 100.0),
        (2, 2, 200.0),
        (3, 3, 300.0),
        (4, 2, 200.0),
        (5, 1, 100.0),
        (6, 2, 200.0),
        (7, 1, 200.0), // <- error
        (8, 2, 300.0), // <- error
        (9, 1, 100.0),
        (10, 2, 200.0),
        (11, 1, 100.0),
        (12, 2, 200.0),
        (13, 1, 100.0),
        (14, 2, 200.0),
        (15, 1, 100.0),
        (16, 2, 200.0)
      )
    ).toDF("uniqueIdColumn", "num", "indexedLabel").toLong("uniqueIdColumn")

    def featuresAssembler = {
      new VectorAssembler()
        .setInputCols(Array("num"))
        .setOutputCol("features")
    }

    val preparedObservations = featuresAssembler.transform(observations)


    val stacking = new SparkGenericStacking(3, "label")

    val Array(trainingSplit,testSplit)  = preparedObservations.randomSplit(Array(0.8, 0.2),11L)

    stacking.foldingStage(trainingSplit, testSplit)

    val problemType = ProblemType.RegressionProblem
    val predictor2 = new LinearRegression()
    stacking.addModel(predictor2, trainingSplit, testSplit, problemType)

    val predictor3 = new LinearRegression().setFitIntercept(false).setRegParam(0.1)
    stacking.addModel(predictor3, trainingSplit, testSplit, problemType)
    val predictor4 = new GBTRegressor()
    stacking.addModel(predictor4, trainingSplit, testSplit, problemType)

    //    stacking.trainModelsPredictionsDF.showAll()
    //    stacking.testModelsPredictionsDF.showAll()

    val finalPredictions = stacking.performStacking(predictor4)
    logger.info("Final predictions GenericStackingShowcaseSuite:")
    //    finalPredictions.showAll()

  }
}


package com

import com.automl.spark.SparkSessionProvider
import com.stacking.GenericStacking
import ml.dmlc.xgboost4j.scala.spark._
import org.apache.spark.ml.{Pipeline, Predictor}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{DenseVector => DenseVectorML, Vector => MLLibVector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.linalg.{Vector => FeatureVector}
import org.apache.spark.sql.DataFrame
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import utils._

object StackingPredictorApp extends SparkSessionProvider{


  def main(args: Array[String]) {

    val path = System.getProperty("user.dir") + s"/"
    val stackingTrainSet = SparkMLUtils.loadAbsoluteDF(path + "stacking_train_price/stacking_price_train.csv")
    val stackingTestSet = SparkMLUtils.loadAbsoluteDF(path + "stacking_test_price/stacking_price_test.csv")


    val evaluator = new RegressionEvaluator()
    /*
    * XGBoost section
    * */
    val xGBoostEstimator = new XGBoostEstimator("PricePredictor")
    val pipeline = new Pipeline().setStages(Array(xGBoostEstimator))

    val paramsMap: Array[ParamMap] = new ParamGridBuilder()
      .addGrid(xGBoostEstimator.eta, Array(0.09))
      .addGrid(xGBoostEstimator.round, Array(170, 250))
      .addGrid(xGBoostEstimator.minChildWeight, Array(1.0))
      .addGrid(xGBoostEstimator.maxDepth, Array(3))
      .addGrid(xGBoostEstimator.subSample, Array(0.7, 0.8))
      .addGrid(xGBoostEstimator.colSampleByTree, Array(0.8, 0.7))
      .addGrid(xGBoostEstimator.objective, Array("reg:linear"))
      .addGrid(xGBoostEstimator.boosterType, Array("gbtree"))
      .addGrid(xGBoostEstimator.evalMetric, Array("rmse"))
      .addGrid(xGBoostEstimator.baseScore, Array(0.5))
      .addGrid(xGBoostEstimator.scalePosWeight, Array(1.0))

      .addGrid(xGBoostEstimator.colSampleByLevel, Array(1.0))
      .addGrid(xGBoostEstimator.missing, Array(Float.NaN))
      .addGrid(xGBoostEstimator.maxDeltaStep, Array(0.0))
      .addGrid(xGBoostEstimator.gamma, Array(0.0))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramsMap)
      .setNumFolds(3)

    val modelsCount = 3
    val metaFeatures = (0 until modelsCount).toArray.map(idx => s"prediction$idx")

    def featuresAssembler = new VectorAssembler()
      .setInputCols(metaFeatures)
      .setOutputCol("features")

    val trainAssembled = featuresAssembler.transform(stackingTrainSet)
    val testAssembled = featuresAssembler.transform(stackingTestSet)

    val cvModel = cv.fit(trainAssembled)
    println("Average metrics:" + cvModel.avgMetrics.sortWith(_ < _).mkString)
    val finalPredictions = cvModel.transform(testAssembled)

    //      finalPredictions.show(100, false)
    //      import ss.implicits._
    //
    //      CSVSaver.saveToCSVWithRewrite(finalPredictions.select($"id", $"prediction".as("price_doc")), "stacking_price_pred")

  }
}







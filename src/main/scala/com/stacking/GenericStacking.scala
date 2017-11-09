package com.stacking

import java.util.{Collections, Random}
import java.util.concurrent.TimeUnit

import ml.dmlc.xgboost4j.scala.spark.XGBoostEstimator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{PipelineStage, Predictor}
import org.apache.spark.ml.linalg.{DenseMatrix, Vector}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator
import org.deeplearning4j.earlystopping.termination.{MaxEpochsTerminationCondition, MaxTimeIterationTerminationCondition, ScoreImprovementEpochTerminationCondition}
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import utils.DeepLearningHelper
import org.apache.spark.ml.linalg.{DenseMatrix, Vector => MLVector}
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import utils.DeepLearningHelper._

import scala.collection.JavaConverters._


//TODO implement Blending https://mlwave.com/kaggle-ensembling-guide/
class GenericStacking(numFold: Int) {

  val seed = 1234

  var splits: Array[(DataFrame, DataFrame)] = _
  var trainModelsPredictionsDF: DataFrame = _
  var testModelsPredictionsDF: DataFrame = _

  /*
  * @df training set which wil be folded
  * */
  def foldingStage(trainDF: DataFrame, testDF: DataFrame) = {

    val projection = trainDF.select("id", "label")
    val ss = trainDF.sparkSession

    /*
    * Making base for future prediction columns appendings
    * */
    trainModelsPredictionsDF = projection.cache()
    testModelsPredictionsDF = testDF.select("id").cache()

    val splitsRdd = MLUtils.kFold(projection.rdd, numFold, seed)
    val schema = new StructType(Array(StructField("id", IntegerType, nullable = false)))
    splits = splitsRdd.map { case (training, validation) =>
      val trainingSplitDF = ss.createDataFrame(training, schema).cache()
      val validationSplitDF = ss.createDataFrame(validation, schema).cache()
      (trainingSplitDF, validationSplitDF)
    }
  }

  private var modelsCount = 0

  def addModel(net: MultiLayerNetwork, trainDataSet: DataFrame, testDataSet: DataFrame, withEarlyStoppingByScore:Boolean) = {

    trainDataSet.cache()
    testDataSet.cache()

    import DeepLearningHelper._
    val ss = trainDataSet.sparkSession
    import ss.implicits._

    net.init()
    net.setListeners(new ScoreIterationListener(1))

    val iteratorParams = DNNIteratorParams(
      numInputs = trainDataSet.select("features").first().getAs[MLVector](0).size,
      batchSize = 100
    )

    /*
    * First stage
    * */

    val splitsWithPredictions = splits.zipWithIndex.map { case ((trainingSplitDF, validationSplitDF), splitIndex) =>

      val trainingFold = trainingSplitDF.join(trainDataSet, "id")
      val validationFold = validationSplitDF.join(trainDataSet, "id")

      val trainer = new EarlyStoppingWrapper(net, trainingFold, validationFold, iteratorParams).getTrainer

      val trainingResult = trainer.fit()

      printLearingProcessInfo(trainingResult)

      val bestModel: MultiLayerNetwork = trainingResult.getBestModel

      bestModel.transform(validationFold, iteratorParams, s"prediction$modelsCount")
    }
    val reunitedSplits: DataFrame = splitsWithPredictions.reduceLeft((acc, next) => acc.union(next)).drop("label")
    trainModelsPredictionsDF = trainModelsPredictionsDF.join(reunitedSplits, "id").cache()

    /*
    * Second stage
    * */

    val trainer =
      if (withEarlyStoppingByScore) {
        new EarlyStoppingWrapper(net, trainDataSet, testDataSet, iteratorParams).getTrainer
      }
      else new EarlyStoppingWrapper(net, trainDataSet, testDataSet, iteratorParams, withoutScoreTermination = true).getTrainer

    val trainingResult = trainer.fit()
    printLearingProcessInfo(trainingResult)
    val bestModel: MultiLayerNetwork = trainingResult.getBestModel


    val testPredictions = bestModel.transform(testDataSet, iteratorParams, s"prediction$modelsCount")
    testModelsPredictionsDF = testModelsPredictionsDF.join(testPredictions, "id").cache()

    modelsCount += 1

    this
  }

  def addModel(predictor: PipelineStage,  trainDataSet: DataFrame, testDataSet: DataFrame) = {

    trainDataSet.cache()
    testDataSet.cache()

    val predictionCol: String = s"prediction$modelsCount"

    /*
    * First stage
    * */

    val splitsWithPredictions = splits.zipWithIndex.map { case ((trainingSplitDF, validationSplitDF), splitIndex) =>

      val trainingFold = trainingSplitDF.join(trainDataSet, "id").cache()
      val validationFold = validationSplitDF.join(trainDataSet, "id")

      predictor match {
        case pr: Predictor[_,_,_] =>
          val model = pr.fit(trainingFold)
          model.setPredictionCol(predictionCol)
          val predictions = model.transform(validationFold)
          predictions.cache()
      }

    }
    val reunitedSplits: DataFrame = splitsWithPredictions.reduceLeft((acc, next) => acc.union(next)).select("id", predictionCol)

    trainModelsPredictionsDF = trainModelsPredictionsDF.join(reunitedSplits, "id").cache()

    /*
    * Second stage
    * */
    val predictionsForTestSetDF: DataFrame = (predictor match {
      case pr: Predictor[_,_,_] =>
        val model = pr.fit(trainDataSet)
        model.setPredictionCol(predictionCol)
        val predictions = model.transform(testDataSet)
        predictions.cache()
    }).select("id", predictionCol)

    testModelsPredictionsDF = testModelsPredictionsDF.join(predictionsForTestSetDF, "id").cache()

    modelsCount += 1
    this
  }


  def performStacking(metaPredictor: PipelineStage) = {
    val metaFeatures = (0 until modelsCount).toArray.map(idx => s"prediction$idx")
    def featuresAssembler = new VectorAssembler()
      .setInputCols(metaFeatures)
      .setOutputCol("features")

    val trainAssembled = featuresAssembler.transform(trainModelsPredictionsDF)
    val testAssembled = featuresAssembler.transform(testModelsPredictionsDF)

    val metaModel = metaPredictor match {
      case pr: Predictor[_, _, _] => pr.fit(trainAssembled)
    }
    metaModel.transform(testAssembled)
  }

}


class EarlyStoppingWrapper(net: MultiLayerNetwork, trainingSet: DataFrame, validationSet: DataFrame, params: DNNIteratorParams, withoutScoreTermination: Boolean = false) {

  val seed = 4567
  val rnd = new Random(seed)

  val modelsDir = System.getProperty("user.dir")+"/src/main/resources/epochModels/"

  def getTrainer = {

    val esConf =
      if(withoutScoreTermination) {
        new EarlyStoppingConfiguration.Builder()
          .epochTerminationConditions(new MaxEpochsTerminationCondition(500))
          .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(90, TimeUnit.MINUTES))
          .evaluateEveryNEpochs(1)
          .modelSaver(new LocalFileModelSaver(modelsDir))
          .build()
      }
      else {
        val validationSetIterator = trainingIteratorFromDF(validationSet, params)
        new EarlyStoppingConfiguration.Builder()
          .epochTerminationConditions(Array(new MaxEpochsTerminationCondition(500),
            new ScoreImprovementEpochTerminationCondition(10)).toList.asJava)
          .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(90, TimeUnit.MINUTES))
          .scoreCalculator(new DataSetLossCalculator(validationSetIterator, true))
          .evaluateEveryNEpochs(1)
          .modelSaver(new LocalFileModelSaver(modelsDir))
          .build()
      }


    val trainingIterator = trainingIteratorFromDF(trainingSet, params)

    new EarlyStoppingTrainer(esConf, net, trainingIterator)

  }
}

case class DNNIteratorParams(numInputs: Int, batchSize: Int)
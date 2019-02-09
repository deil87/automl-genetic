package com.automl.classifier.ensemble.stacking

import java.util.Random
import java.util.concurrent.TimeUnit

import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.problemtype.ProblemType
import com.automl.template.{TemplateMember, TemplateTree}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.classification.Classifier
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.ml.{PipelineStage, Predictor}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.{IntegerType, LongType, StructField, StructType}
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator
import org.deeplearning4j.earlystopping.termination.{MaxEpochsTerminationCondition, MaxTimeIterationTerminationCondition, ScoreImprovementEpochTerminationCondition}
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import utils.DeepLearningHelper
import utils.DeepLearningHelper._

import scala.collection.JavaConverters._


//TODO implement Blending https://mlwave.com/kaggle-ensembling-guide/
class SparkGenericStacking(numFold: Int, responseColumn: String) extends LazyLogging{

  val seed = 1234

  var splits: Array[(DataFrame, DataFrame)] = _
  var trainModelsPredictionsDF: DataFrame = _
  var testModelsPredictionsDF: DataFrame = _

  /*
  * @df training set which wil be folded
  * */
  def foldingStage(trainDF: DataFrame, testDF: DataFrame) = {
    logger.debug(s"Folding stage of SparkGenericStacking is started. Preparing $numFold train/valid pair from training frame.")

    val projection = trainDF.select("uniqueIdColumn", responseColumn)
    val ss = trainDF.sparkSession

    /*
    * Making base for future prediction columns appendings
    * */
    trainModelsPredictionsDF = projection.cache()
    testModelsPredictionsDF = testDF.select("uniqueIdColumn").cache()

    val splitsRdd = MLUtils.kFold(projection.rdd, numFold, seed)
    val schema = new StructType(Array(StructField("uniqueIdColumn", LongType, nullable = false)))
    splits = splitsRdd.map { case (training, validation) =>
      val trainingSplitDF = ss.createDataFrame(training, schema).cache()
      val validationSplitDF = ss.createDataFrame(validation, schema).cache()
      logger.debug(f"Created ${training.count()}%10s train / ${validation.count()}%-10s validation pair.")
      (trainingSplitDF, validationSplitDF)
    }
  }

  private var numberOfModels = 0

  /**
    * Adding MultiLayerNetwork to the ensemble.
    */
  def addModel(net: MultiLayerNetwork, trainDataSet: DataFrame, testDataSet: DataFrame, withEarlyStoppingByScore:Boolean) = {

    trainDataSet.cache()
    testDataSet.cache()

    import DeepLearningHelper._
    val ss = trainDataSet.sparkSession

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

      val trainingFold = trainingSplitDF.join(trainDataSet, "uniqueIdColumn")
      val validationFold = validationSplitDF.join(trainDataSet, "uniqueIdColumn")

      val trainer = new EarlyStoppingWrapper(net, trainingFold, validationFold, iteratorParams).getTrainer

      val trainingResult = trainer.fit()

      printLearningProcessInfo(trainingResult)

      val bestModel: MultiLayerNetwork = trainingResult.getBestModel

      bestModel.transform(validationFold, iteratorParams, s"prediction$numberOfModels")
    }
    val reunitedSplits: DataFrame = splitsWithPredictions.reduceLeft((acc, next) => acc.union(next)).drop(responseColumn)
    trainModelsPredictionsDF = trainModelsPredictionsDF.join(reunitedSplits, "uniqueIdColumn").cache()

    /*
    * Second stage
    * */

    val trainer =
      if (withEarlyStoppingByScore) {
        new EarlyStoppingWrapper(net, trainDataSet, testDataSet, iteratorParams).getTrainer
      }
      else new EarlyStoppingWrapper(net, trainDataSet, testDataSet, iteratorParams, withoutScoreTermination = true).getTrainer

    val trainingResult = trainer.fit()
    printLearningProcessInfo(trainingResult)
    val bestModel: MultiLayerNetwork = trainingResult.getBestModel


    val testPredictions = bestModel.transform(testDataSet, iteratorParams, s"prediction$numberOfModels")
    testModelsPredictionsDF = testModelsPredictionsDF.join(testPredictions, "uniqueIdColumn").cache()

    numberOfModels += 1

    this
  }

  /**
    * Adding TemplateMember to the ensemble
    */
  def addModel[A <: TemplateMember](member: TemplateTree[A], trainDataSet: DataFrame, testDataSet: DataFrame, problemType: ProblemType, hyperParamsMap: HyperParametersField): SparkGenericStacking = {
    trainDataSet.cache()
    testDataSet.cache()
    import utils.SparkMLUtils._
    import trainDataSet.sparkSession.implicits._

    val predictionCol: String = s"prediction$numberOfModels"

    /*
    * First stage
    * */
    val numberOfFolds = splits.length

    val splitsWithPredictions = splits.zipWithIndex.map { case ((trainingFoldIds, holdoutFoldIds), splitIndex) =>

      val trainingFold = trainingFoldIds.join(trainDataSet, "uniqueIdColumn").cache()
      val holdoutFold = holdoutFoldIds.join(trainDataSet, "uniqueIdColumn").cache()

      val fitnessResultWithPredictions = member.evaluateFitness(trainingFold, holdoutFold, problemType, hyperParamsMap)

      val holdoutFoldPredictions = fitnessResultWithPredictions.dfWithPredictions
        .withColumnRenamed("prediction", predictionCol)
        .cache()
//      if(splitIndex == 1)
//        holdoutFoldPredictions.withColumnReplace(predictionCol, $"$predictionCol" + new Random().nextDouble()*100)
//      else
      holdoutFoldPredictions
    }

    val reunitedSplits: DataFrame = splitsWithPredictions
      .reduceLeft((reunitedDataFrame, next) => reunitedDataFrame.union(next))
      .select("uniqueIdColumn", predictionCol)

    require(trainDataSet.count() == reunitedSplits.count(), "Reunited splits do not sum up to the original training dataset's size.")

    logger.info(s"CrossValidated predictions( nfolds = $numberOfFolds ) from ${member.member.name} were reunited and joined to the `trainModelsPredictionsDF`")

    trainModelsPredictionsDF = trainModelsPredictionsDF.join(reunitedSplits, "uniqueIdColumn").cache()

    /*
    * Second stage
    * */

    logger.info(s"Predictions for test split are going to be calculateg by ${member.member.name} model trained on whole training dataset")

    val predictionsForTestSetDF: DataFrame =
      member
        .evaluateFitness(trainDataSet, testDataSet, problemType, hyperParamsMap)
        .dfWithPredictions
        .withColumnRenamed("prediction", predictionCol)
//      .showN_AndContinue(100, "Raw predictions")
        .select("uniqueIdColumn", predictionCol)

    logger.info(s"Predictions based on the whole training dataset were calculated by ${member.member.name} and added to `testModelsPredictionsDF`")

    testModelsPredictionsDF = testModelsPredictionsDF.join(predictionsForTestSetDF, "uniqueIdColumn").cache()

    import testDataSet.sparkSession.implicits._
    import org.apache.spark.sql.functions._

//    trainModelsPredictionsDF.select("uniqueIdColumn", predictionCol).join(trainDataSet.select("uniqueIdColumn", "indexedLabel").withColumnRenamed("indexedLabel", "indexedLabel_train"), "uniqueIdColumn")
//        .withColumn("diff", when($"$predictionCol" === $"indexedLabel_train", 1).otherwise(0))
//        .showN_AndContinue(100, "Diff show")


    numberOfModels += 1
    this

  }

    def addModel(predictor: PipelineStage,  trainDataSet: DataFrame, testDataSet: DataFrame, problemType: ProblemType) = {

    trainDataSet.cache()
    testDataSet.cache()

    val predictionCol: String = s"prediction$numberOfModels"

    /*
    * First stage
    * */

    val splitsWithPredictions = splits.zipWithIndex.map { case ((trainingSplitDF, validationSplitDF), splitIndex) =>

      val trainingFold = trainingSplitDF.join(trainDataSet, "uniqueIdColumn").cache()
      val validationFold = validationSplitDF.join(trainDataSet, "uniqueIdColumn")

      predictor match {
        case pr: Predictor[_,_,_] =>
          val model = pr.fit(trainingFold)
          model.setPredictionCol(predictionCol)
          val predictions = model.transform(validationFold)
          predictions.cache()
      }

    }
    val reunitedSplits: DataFrame = splitsWithPredictions.reduceLeft((acc, next) => acc.union(next))
      .select("uniqueIdColumn", predictionCol)

    trainModelsPredictionsDF = trainModelsPredictionsDF.join(reunitedSplits, "uniqueIdColumn").cache()

    /*
    * Second stage
    * */
    val predictionsForTestSetDF: DataFrame = (predictor match {
      case pr: Predictor[_,_,_] =>
        val model = pr.fit(trainDataSet)
        model.setPredictionCol(predictionCol)
        val predictions = model.transform(testDataSet)
        predictions.cache()
    }).select("uniqueIdColumn", predictionCol)

    testModelsPredictionsDF = testModelsPredictionsDF.join(predictionsForTestSetDF, "uniqueIdColumn").cache()

    numberOfModels += 1
    this
  }


  def performStacking(metaPredictor: PipelineStage) = {
    import utils.SparkMLUtils._
    val metaFeatures = (0 until numberOfModels).toArray.map(idx => s"prediction$idx")
    def featuresAssembler = new VectorAssembler()
      .setInputCols(metaFeatures)
      .setOutputCol("features")

    val trainAssembled = featuresAssembler.transform(trainModelsPredictionsDF).drop(metaFeatures:_*)
    val testAssembled = featuresAssembler.transform(testModelsPredictionsDF).drop(metaFeatures:_*)

//    trainAssembled.showN_AndContinue(trainAssembled.count().toInt, "Perform fitting of the metalearner")

    val metaModel = metaPredictor match {
      case pr: Predictor[_, _, _] => pr.fit(trainAssembled) // .setLabelCol("indexedLabel") is done on the caller's side
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
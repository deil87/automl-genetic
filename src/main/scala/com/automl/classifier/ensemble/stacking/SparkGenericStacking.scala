package com.automl.classifier.ensemble.stacking

import java.util.Random
import java.util.concurrent.TimeUnit

import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.problemtype.ProblemType
import com.automl.template.{TemplateMember, TemplateTree}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.classification.Classifier
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector => MLVector}
import org.apache.spark.ml.{PipelineStage, Predictor}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, lit}
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
    logger.debug(s"Folding stage of SparkGenericStacking is started. Preparing $numFold train/valid pairs based on training frame( rows=${trainDF.count()}).")

    val projection = trainDF.select("uniqueIdColumn", responseColumn)

    val ss = trainDF.sparkSession

    /*
    * Making base for future prediction columns appendings
    * */
    trainModelsPredictionsDF = projection.cache()
    testModelsPredictionsDF = testDF.select("uniqueIdColumn").cache()

    val splitsRdd = MLUtils.kFold(projection.rdd, numFold, seed + projection.count())
    val schema = new StructType(Array(StructField("uniqueIdColumn", LongType, nullable = false)))
    //converting rdds -> DFs
    splits = splitsRdd.map { case (training, validation) =>
      val validationCount = validation.count()
      val trainCount = training.count()
      val trainingSplitDF = ss.createDataFrame(training, schema).cache()
      val validationSplitDF = ss.createDataFrame(validation, schema).cache()
      logger.debug(f"Created ${trainCount}%10s train / ${validationCount}%-10s validation pair.")
      require(validationCount > 0 && trainCount > 0, "Validation or training split is of 0 size")
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
  def addModel[A <: TemplateMember](member: TemplateTree[A], trainDataSet: DataFrame, testDataSet: DataFrame, problemType: ProblemType, hyperParamsMap: Option[HyperParametersField]): SparkGenericStacking = {
    logger.info(s"Adding TemplateMember ${member.render} to the ensemble")
    trainDataSet.cache()
    testDataSet.cache()
    import utils.SparkMLUtils._
    import trainDataSet.sparkSession.implicits._
    import org.apache.spark.ml.linalg.DenseVector

    val predictionCol: String = s"prediction$numberOfModels"
    val weightCol: String = s"weight$numberOfModels"
    val probabilityCol: String = s"probability$numberOfModels"
    val weightedProbCol: String = s"weightedProb$numberOfModels"

    def weightedProbability = {
      import org.apache.spark.sql.functions.udf
      udf { (components: DenseVector, weight: Double) =>
        new DenseVector(components.values.map(v => v * weight))
      }
    }
    /*
    * First stage
    * */
    val numberOfFolds = splits.length

    val splitsWithPredictions = splits.zipWithIndex.map { case ((trainingFoldIds, holdoutFoldIds), splitIndex) =>

      val trainingFold = trainingFoldIds.join(trainDataSet, "uniqueIdColumn").cache()
      val holdoutFold = holdoutFoldIds.join(trainDataSet, "uniqueIdColumn").cache()

      val fitnessResultWithPredictions = member.evaluateFitness(trainingFold, holdoutFold, problemType, hyperParamsMap)

      val holdoutFoldPredictions = fitnessResultWithPredictions.dfWithPredictions
        .withColumnRenamed("prediction", predictionCol) // we can use here `rawPrediction` and `probability` columns.
        .withColumnRenamed("probability", probabilityCol) // we can use here `rawPrediction` and `probability` columns.
        .withColumn(weightCol, lit(fitnessResultWithPredictions.getCorrespondingMetric))
        .withColumn(weightedProbCol, weightedProbability(col(probabilityCol), lit(fitnessResultWithPredictions.getCorrespondingMetric)))
//        .showN_AndContinue(500, "Splits predictions:")
        .cache()

      holdoutFoldPredictions
    }

    val reunitedSplits: DataFrame = splitsWithPredictions
      .reduceLeft((reunitedDataFrame, next) => reunitedDataFrame.union(next))
      .select("uniqueIdColumn", weightedProbCol)
//        .showAllAndContinue

    require(trainDataSet.count() == reunitedSplits.count(), "Reunited splits do not sum up to the original training dataset's size.")

    logger.info(s"CrossValidated predictions( nfolds = $numberOfFolds ) from ${member.member.name} were reunited and joined to the `trainModelsPredictionsDF`")

    trainModelsPredictionsDF = trainModelsPredictionsDF.join(reunitedSplits, "uniqueIdColumn").cache()

    /*
    * Second stage
    * */

    logger.info(s"Predictions for test split are going to be calculated by ${member.member.name} model trained on whole training dataset")

    val fitnessResultOnWholeData = member
      .evaluateFitness(trainDataSet, testDataSet, problemType, hyperParamsMap)

    val predictionsForTestSetDF: DataFrame =
      fitnessResultOnWholeData
        .dfWithPredictions
        .withColumnRenamed("prediction", predictionCol)
        .withColumn(weightedProbCol, weightedProbability(col("probability"), lit(fitnessResultOnWholeData.getCorrespondingMetric)))
//        .showN_AndContinue(500, "Test weighted probs")
        .select("uniqueIdColumn", predictionCol, weightedProbCol)

    logger.info(s"Predictions based on the whole training dataset were calculated by ${member.member.name} and added to `testModelsPredictionsDF`")

    testModelsPredictionsDF = testModelsPredictionsDF.join(predictionsForTestSetDF, "uniqueIdColumn").cache()

    logger.info(s"Predictions based on the whole training dataset were calculated and joined")

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
    val metaFeaturesWP = (0 until numberOfModels).toArray.map(idx => s"weightedProb$idx")
    def featuresAssembler = new VectorAssembler()
      .setInputCols(metaFeatures)
      .setOutputCol("features")

    def weightedProbabilityCombine = {
      import org.apache.spark.sql.functions.udf
      udf { (weightedProb1: DenseVector, weightedProb2: DenseVector) =>
        new DenseVector(weightedProb1.toArray.zip(weightedProb2.toArray).map{case (a,b)=> a + b})
      }
    }
    def emptyArray = {
      import org.apache.spark.sql.functions.udf
      udf { size: Int =>
        new DenseVector(Array.fill(size)(0))
      }
    }

    val numClasses = trainModelsPredictionsDF.select("indexedLabel").distinct().count().toInt // TODO it might not be the case!!!! pass num of classes in another way
    val trainModelsPredictionsDFWithAccumulator = trainModelsPredictionsDF.withColumn("weightedProbSoftVoted", emptyArray(lit(numClasses)))
    val combinedTrainWeightedProabilities = metaFeaturesWP.foldLeft(trainModelsPredictionsDFWithAccumulator)((res, next) =>
      res.withColumnReplace("weightedProbSoftVoted", weightedProbabilityCombine(col("weightedProbSoftVoted"), col(next)))
    )


    val trainAssembled = combinedTrainWeightedProabilities
      .withColumnRenamed("weightedProbSoftVoted", "features") // we might need to divide each element by number of predictors
      .drop(metaFeatures:_*)
      .drop(metaFeaturesWP:_*)

//    trainAssembled.showN_AndContinue(50, "All combined with foldLeft metafeatures of the train dataset")

    val testModelsPredictionsDFWithAccumulator = testModelsPredictionsDF.withColumn("weightedProbSoftVoted", emptyArray(lit(numClasses)))
    val combinedTestWeightedProabilities = metaFeaturesWP.foldLeft(testModelsPredictionsDFWithAccumulator)((res, next) =>
      res.withColumnReplace("weightedProbSoftVoted", weightedProbabilityCombine(col("weightedProbSoftVoted"), col(next)))
    )

    val testAssembled = combinedTestWeightedProabilities
      .withColumnRenamed("weightedProbSoftVoted", "features")
      .drop(metaFeatures:_*)
      .drop(metaFeaturesWP:_*)

//    testAssembled.showN_AndContinue(50, "All weighted probabilities combined with foldLeft  of the test dataset")


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
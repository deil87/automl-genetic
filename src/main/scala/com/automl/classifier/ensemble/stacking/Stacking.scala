package com.automl.classifier.ensemble.stacking

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{PipelineStage, Predictor}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.DataFrame


class Stacking(numFold: Int, withFeatures: Boolean = false) {

  var stagePredictors:Array[PipelineStage] = null
  var metaPredictor : PipelineStage = null

  val seed = 1234

  def setPredictors(value: Array[_ <: PipelineStage]) = {
    stagePredictors = value.asInstanceOf[Array[PipelineStage]]
    this
  }

  def setMetaPredictor(value: PipelineStage) = {
    metaPredictor = value.asInstanceOf[PipelineStage]
    this
  }

  /*
  *  Leave-one-out-fold transformation
  * */
  def firstStage(df: DataFrame) = {
    val schema = df.schema
    val sparkSession = df.sparkSession
    import org.apache.spark.sql.functions._

    val splits = MLUtils.kFold(df.rdd, numFold, seed)
    val splitsWithPredictions =
      splits.zipWithIndex.map { case ((training, validation), splitIndex) =>
        val trainingDataset = sparkSession.createDataFrame(training, schema).cache()
        val testDataset = sparkSession.createDataFrame(validation, schema).cache()

        val withFoldIdTestSet = testDataset.withColumn("foldId", lit(splitIndex))

        stagePredictors.zipWithIndex.foldLeft(withFoldIdTestSet) { case (testSetAcc, (predictor, index)) =>
          predictor match {
            case pr: Predictor[_,_,_] =>
              val model = pr.fit(trainingDataset)
              model.setPredictionCol(s"prediction$index")
              val predictions = model.transform(testSetAcc)
              predictions
          }

        }

      }

    val reunitedSplits: DataFrame = splitsWithPredictions.reduceLeft((acc, next) => acc.union(next))
    reunitedSplits

  }

  /*
  *  Fit with all training data and predict for testSet
  * */
  def secondStage(trainingSet: DataFrame, testSet: DataFrame) = {
    trainingSet.cache()
    testSet.cache()
    stagePredictors.zipWithIndex.foldLeft(testSet) { case (testSetAcc, (predictor, index)) =>
      predictor match {
        case pr: Predictor[_,_,_] =>
          val model = pr.fit(trainingSet)
          model.setPredictionCol(s"prediction$index")
          val predictions = model.transform(testSetAcc)
          predictions
      }
    }

  }

  /*
  *  ???
  * */
  def finalStage(trainingTransformedSet: DataFrame, testTransformedSet: DataFrame) = {

    val metaFeatures = stagePredictors.indices.toArray.map(idx => s"prediction$idx")
    def basePredictorsFeaturesAssembler = new VectorAssembler()
      .setInputCols(metaFeatures)
      .setOutputCol("features")

    /*
    * Dropping previous features column so that we can assemble out first-stage and second-stage predictions into new features column.
    * */
    val trainAssembled = basePredictorsFeaturesAssembler.transform(trainingTransformedSet.drop("features"))
    val testAssembled = basePredictorsFeaturesAssembler.transform(testTransformedSet.drop("features"))

    val metaModel = metaPredictor match {
        case pr: Predictor[_, _, _] => pr.fit(trainAssembled)
      }

      val predictions = metaModel.transform(testAssembled)
      predictions

  }

}

package com.stacking

import com.automl.spark.SparkSessionProvider
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{GBTRegressor, LinearRegression}
import org.scalatest.{FunSuite, Matchers}


class GenericStackingSuite extends FunSuite with Matchers with SparkSessionProvider with LazyLogging{

  import ss.implicits._
  import utils.SparkMLUtils._

  test("Should combine models for stacking") {
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
    ).toDF("id", "num", "label")

    def featuresAssembler = {
      new VectorAssembler()
        .setInputCols(Array("num"))
        .setOutputCol("features")
    }

    val preparedObservations = featuresAssembler.transform(observations)


    val stacking = new SparkGenericStacking(3)

    val Array(trainingSplit,testSplit)  = preparedObservations.randomSplit(Array(0.8, 0.2),11L)

    stacking.foldingStage(trainingSplit, testSplit)

    val predictor2 = new LinearRegression()
    stacking.addModel(predictor2, trainingSplit, testSplit)

    val predictor3 = new LinearRegression().setFitIntercept(false).setRegParam(0.1)
    stacking.addModel(predictor3, trainingSplit, testSplit)
    val predictor4 = new GBTRegressor()
    stacking.addModel(predictor4, trainingSplit, testSplit)

    stacking.trainModelsPredictionsDF.showAll()
    stacking.testModelsPredictionsDF.showAll()

    val finalPredictions = stacking.performStacking(predictor4)
    logger.info("Final predictions:")
    finalPredictions.showAll()

  }

}


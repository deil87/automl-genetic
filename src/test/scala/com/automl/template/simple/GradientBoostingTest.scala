package com.automl.template.simple

import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem}
import com.automl.spark.SparkSessionProvider
import org.apache.spark.ml.feature.VectorAssembler
import org.scalatest.{FunSuite, Ignore}
import utils.SparkMLUtils

class GradientBoostingTest extends FunSuite with SparkSessionProvider{

  import utils.SparkMLUtils._

  test("testFitnessError method should return note that GBTClassifier currently only supports binary classification ") {

    val data = SparkMLUtils.loadResourceDF("/iris.csv")

    val preparedData = data
      .withColumnRenamed("fl_class", "label")

    def basePredictorsFeaturesAssembler = new VectorAssembler()
      .setInputCols(Array("s_length", "s_width", "p_length", "p_width"))
      .setOutputCol("features")

    val featuresAssembled = basePredictorsFeaturesAssembler.transform(preparedData)

    val Array(trainingSplit, testSplit) = featuresAssembled.randomSplit(Array(0.67, 0.33), 11L)
    val res: FitnessResult = GradientBoosting().fitnessError(trainingSplit, testSplit, problemType = MultiClassClassificationProblem, None)
    res.metricsMap

  }

  test("testFitnessError method should return note that GBTClassifier currently only supports binary classification test 2") {

    val data = SparkMLUtils.loadResourceDF("/iris.csv")

    val preparedData = data
      .withColumnRenamed("fl_class", "label")

    def basePredictorsFeaturesAssembler = new VectorAssembler()
      .setInputCols(Array("s_length", "s_width", "p_length", "p_width"))
      .setOutputCol("features")

    val featuresAssembled = basePredictorsFeaturesAssembler.transform(preparedData)

    val Array(trainingSplit, testSplit) = featuresAssembled.randomSplit(Array(0.67, 0.33), 11L)
    val res: FitnessResult = GradientBoosting().fitnessError(trainingSplit, testSplit, problemType = BinaryClassificationProblem, None)
    res.metricsMap

    // TODO transform multiclass task to a binary one or choose another dataset for test.

  }

}

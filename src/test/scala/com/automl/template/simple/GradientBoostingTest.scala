package com.automl.template.simple

import com.automl.ConfigProvider
import com.automl.dataset.Datasets
import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem}
import com.automl.spark.SparkSessionProvider
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.{coalesce, lit, mean, when}
import org.scalatest.{FunSuite, Ignore}
import utils.SparkMLUtils

class GradientBoostingTest extends FunSuite with SparkSessionProvider{

  import utils.SparkMLUtils._

  ignore("testFitnessError method should return note that GBTClassifier currently only supports binary classification ") {

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
    ConfigProvider.clearOverride.addOverride(
      """
        |evolution {
        |  evaluation {
        |    multiclass.metric = "logloss"
        |  }
        |}
      """)

    val data = Datasets.getIrisDataFrame(1234)
    import data.sparkSession.implicits._

    data.showN_AndContinue(10, "Iris sample")

    val preparedData = data
      .withColumnReplace("indexedLabel",  when($"indexedLabel" === 1.0, lit(0.0))
        .otherwise(1.0))
//      .withColumnRenamed("fl_class", "label")

    preparedData.showN_AndContinue(10, "Iris sample")


    val Array(trainingSplit, testSplit) = preparedData.randomSplit(Array(0.67, 0.33), 11L)
    val res: FitnessResult = GradientBoosting().fitnessError(trainingSplit, testSplit, problemType = BinaryClassificationProblem, None)
    res.metricsMap

    // TODO transform multiclass task to a binary one or choose another dataset for test.

  }

}

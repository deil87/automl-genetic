package com.automl.spark.randomforest

import com.automl.ConfigProvider
import com.automl.dataset.Datasets
import com.automl.evolution.dimension.hparameter._
import com.automl.problemtype.ProblemType
import com.automl.spark.SparkSessionProvider
import com.automl.template.simple.RandomForest
import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.scalatest.concurrent.TimeLimitedTests
import org.scalatest.time.{Minutes, Span}
import org.scalatest.{FunSuite, Matchers}
import utils.SparkMLUtils
import com.test.TestBase


class RandomForestSuite extends FunSuite with SparkSessionProvider with TestBase{

  import utils.SparkMLUtils._

  val testOverride: Config = ConfigFactory.parseString(
    """
      |evolution {
      |  hpGridSearch = false
      |  hyperParameterDimension {
      |     enabled = false
      |  }
      |}
    """.stripMargin)
  ConfigProvider.clearOverride.addOverride(testOverride)


  val airlineDF = SparkMLUtils.loadParquet("src/test/resources/airline_allcolumns_sampled_100k_parquet")
    .select("DayOfWeek", "Distance", "DepTime", "CRSDepTime", "DepDelay")

  val features = Array("Distance", "DayOfWeek")
  val oheFeatures = Array.empty

  val combinedFeatures = features

  val featuresColName: String = "features"

  def featuresAssembler = {
    new VectorAssembler()
      .setInputCols(combinedFeatures)
      .setOutputCol(featuresColName)
  }
  import org.apache.spark.sql.functions.monotonically_increasing_id

  val prepairedAirlineDF = airlineDF
    .limit(1000)
    .applyTransformation(featuresAssembler)
    .withColumnRenamed("DepDelay", "label")
    .toDouble("label")
    .filterOutNull("label")
    .withColumn("uniqueIdColumn", monotonically_increasing_id)
//    .showN_AndContinue(100)
    .cache()


  test("Random forest should work with regression problems") {

    val rfr =  new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(prepairedAirlineDF)

    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, rfr))

    val Array(trainingSplit, testSplit) = prepairedAirlineDF.randomSplit(Array(0.8, 0.2))

    trainingSplit.show(10)

      val model = pipeline.fit(trainingSplit)
      val predictions = model.transform(testSplit)

    val evaluator = new RegressionEvaluator()
  }

  ignore("Random forest should work with classification problems") {

    val shuffleSeed = 1234
    val data = Datasets.getBalanceDataFrame(shuffleSeed)

    val Array(trainingSplit, testSplit) = data.randomSplit(Array(0.8, 0.2))

    trainingSplit.show(10)

    val problemType = ProblemType.MultiClassClassificationProblem

    val rfHPFieldOpt = Some(HyperParametersField(
      Seq(
        RandomForestHPGroup(Seq(MaxDepthRF(Some(5.0))))
      )
    ))

    val f1 = RandomForest().fitnessError( trainingSplit, testSplit, problemType, rfHPFieldOpt).getCorrespondingMetric
    println(s"F1 computed for Random Forest model $f1")

    f1 shouldBe 0.8 +- 0.1
  }

  test("mutation of NumTreesRF uses proper step") {

    val hpParameter = NumTreesRF()
    val currentValue = hpParameter.currentValue
    println(currentValue)
    val mutatedHPParameter = hpParameter.mutate()
    println(mutatedHPParameter)
    mutatedHPParameter.currentValue % 10 shouldBe 0

    if(hpParameter.min != hpParameter.max)
      currentValue should not be(mutatedHPParameter.currentValue)

  }


}


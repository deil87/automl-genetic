package com.automl.sandbox

import com.automl.spark.SparkSessionProvider
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, LabeledPoint => MLLabeledPoint}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpec}
import utils.SparkMLUtils

class AnalysisOfAirlineDatasetSuite extends WordSpec with Matchers with BeforeAndAfterAll with SparkSessionProvider {

  import utils.SparkMLUtils._

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

  val preparedAirlineDF = airlineDF
    .applyTransformation(featuresAssembler)
    .withColumnRenamed("DepDelay", "label")
    .toDouble("label")
    .filterOutNull("label")
    .printSchema_AndContinue
    .withColumn("uniqueIdColumn", monotonically_increasing_id)
    .showN_AndContinue(10)
    .cache()


  val Array(trainingSplit, testSplit) = preparedAirlineDF.randomSplit(Array(0.8, 0.2))

  trainingSplit.cache()

  "Airline dataset" should {

    "have different feature importance that we can extract with Random Forest" in {

      val model2 = new RandomForestRegressor().setNumTrees(100)
      val randomForestRegressionModel = model2.fit(trainingSplit)
      randomForestRegressionModel.featureImportances.toArray.zipWithIndex
        .map(_.swap)
        .sortBy(-_._2)
        .foreach(x => println(x._1 + " -> " + x._2))

      // Make predictions.
      val predictionsWithLabels = randomForestRegressionModel.transform(testSplit)

      val evaluator = new RegressionEvaluator()

      val rmse = evaluator.evaluate(predictionsWithLabels)
      println("RMSE = " + rmse)

    }
  }
}

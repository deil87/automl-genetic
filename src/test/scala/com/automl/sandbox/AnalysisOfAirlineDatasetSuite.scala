package com.automl.sandbox

import com.automl.spark.SparkSessionProvider
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, LabeledPoint => MLLabeledPoint}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpec}
import utils.{FeatureImportanceHelper, SparkMLUtils}

class AnalysisOfAirlineDatasetSuite extends WordSpec with Matchers with BeforeAndAfterAll with SparkSessionProvider {

  import utils.SparkMLUtils._

  val airlineDF_allColumns = SparkMLUtils.loadParquet("src/test/resources/airline_allcolumns_sampled_100k_parquet")
  val airlineDF = airlineDF_allColumns.select("UniqueCarrier", "Month", "DayOfWeek", "Distance", "DepTime", "CRSDepTime", "DepDelay")

  airlineDF_allColumns.show(10)

  val features = Array("Distance", "DayOfWeek")
  val oheFeatures = Array("UniqueCarrier_ohe", "Month_ohe")

  val combinedFeatures = features ++ oheFeatures

  val featuresColName: String = "features"

  def featuresAssembler = {
    new VectorAssembler()
      .setInputCols(combinedFeatures)
      .setOutputCol(featuresColName)
  }
  import org.apache.spark.sql.functions.monotonically_increasing_id

  import ss.implicits._

  val preparedAirlineDF = airlineDF
    .indexedAndOHESet(Set("UniqueCarrier"))
    .withColumnReplace("Month", $"Month" - 1 ) // For OHE we need to have zero indexed categories
    .ohEncodedSet(Set("Month"))
    .applyTransformation(featuresAssembler)
    .withColumnRenamed("DepDelay", "label")
    .toDouble("label")
    .filterOutNull("label")
    .printSchema_AndContinue
    .withColumn("uniqueIdColumn", monotonically_increasing_id)
    .showN_AndContinue(50)
    .cache()


  val Array(trainingSplit, testSplit) = preparedAirlineDF.randomSplit(Array(0.8, 0.2))

  trainingSplit.cache()

  "Airline dataset" should {

    "have different feature importance that we can extract with Random Forest" in {

      val model2 = new RandomForestRegressor().setNumTrees(100)
      val randomForestRegressionModel = model2.fit(trainingSplit)
      val indexToFeatureImportanceMap: Array[(Int, Double)] =
        randomForestRegressionModel.featureImportances.toArray
          .zipWithIndex
          .map(_.swap)

      val sortedColumnNamesForFeatures: Array[String] = trainingSplit.computeColumnNamesForFeaturesVector(combinedFeatures)
      FeatureImportanceHelper.printFeatureScores(indexToFeatureImportanceMap, sortedColumnNamesForFeatures) //TODO consider refactoring.

      // Make predictions.
      val predictionsWithLabels = randomForestRegressionModel.transform(testSplit)

      val evaluator = new RegressionEvaluator()

      val rmse = evaluator.evaluate(predictionsWithLabels)
      println("RMSE = " + rmse)

    }
  }
}
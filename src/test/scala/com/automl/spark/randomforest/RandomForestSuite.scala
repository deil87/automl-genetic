package com.automl.spark.randomforest

import com.automl.spark.SparkSessionProvider
import com.automl.template._
import com.automl.template.ensemble.bagging.Bagging
import com.automl.template.simple._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.scalatest.{FunSuite, Matchers}
import utils.SparkMLUtils


class RandomForestSuite extends FunSuite with Matchers with SparkSessionProvider{


  import utils.SparkMLUtils._

  val airlineDF = SparkMLUtils.loadResourceDF("/airline2008-2.csv")
    .select("DayOfWeek", "Distance", "DepTime", "CRSDepTime", "DepDelay")
  //TODO FlightNum+year_date_day for unique identifier of test examples

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
    .showN_AndContinue(100)
    .cache()


  test("Random forest should calculate over two base models") {

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


}


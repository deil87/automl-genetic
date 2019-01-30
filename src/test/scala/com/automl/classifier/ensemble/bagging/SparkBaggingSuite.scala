package com.automl.classifier.ensemble.bagging

import com.automl.helper.TemplateTreeHelper
import com.automl.problemtype.ProblemType
import com.automl.spark.SparkSessionProvider
import com.automl.template._
import com.automl.template.simple._
import org.apache.spark.ml.feature.VectorAssembler
import org.scalatest.{FunSuite, Matchers}
import utils.SparkMLUtils


class SparkBaggingSuite extends FunSuite with Matchers with SparkSessionProvider{

  import utils.SparkMLUtils._

  val airlineDF = SparkMLUtils.loadParquet("src/test/resources/airline_allcolumns_sampled_100k_parquet")
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
    .limit(3000)
    .applyTransformation(featuresAssembler)
    .withColumnRenamed("DepDelay", "label")
    .toDouble("label")
    .filterOutNull("label")
    .withColumn("uniqueIdColumn", monotonically_increasing_id)
    .showN_AndContinue(100)
    .cache()


  test("Spark Bagging should calculate over complex tree algorithm") {

    val models = Seq(
      LeafTemplate(new LinearRegressionModel()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(RandomForest()),
      LeafTemplate(Bayesian()),
      NodeTemplate(SparkBagging(), Seq(
        LeafTemplate(new LinearRegressionModel()),
        LeafTemplate(GradientBoosting()),
        LeafTemplate(DecisionTree())
      ))
    )

    val ensemb = NodeTemplate(SparkBagging(), models)

    println(TemplateTreeHelper.renderAsString_v2(ensemb))

    val Array(trainingSplit, testSplit) = prepairedAirlineDF.randomSplit(Array(0.8, 0.2))

    ensemb.evaluateFitness(trainingSplit, testSplit, ProblemType.RegressionProblem)
  }

  test("Spark Bagging should calculate over complex tree algorithm( Classification problem )") {

    val models = Seq(
      LeafTemplate(DecisionTree()), //TODO We need n-classes +1 base models to be able to find majority
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree())
    )

    val ensemb = NodeTemplate(SparkBagging(), models)

    println(TemplateTreeHelper.renderAsString_v2(ensemb))

    val data = SparkMLUtils.loadResourceDF("/iris.csv")

    val preparedData = data
      .withColumnRenamed("fl_class", "label")
      .withColumn("uniqueIdColumn", monotonically_increasing_id)

    def basePredictorsFeaturesAssembler = new VectorAssembler()
      .setInputCols(Array("s_length", "s_width", "p_length", "p_width"))
      .setOutputCol("features")

    val featuresAssembled = basePredictorsFeaturesAssembler.transform(preparedData)

    val Array(trainingSplit, testSplit) = featuresAssembled.randomSplit(Array(0.67, 0.33), 11L)

    val problemType = ProblemType.MultiClassClassificationProblem

    val baggingF1 = ensemb.evaluateFitness(trainingSplit, testSplit, problemType).getCorrespondingMetric

    val dtF1 = DecisionTree().fitnessError(trainingSplit, testSplit, problemType).getMetricByName("accuracy")

    println("Bagging's F1:" + baggingF1)
    println("DT's F1:" + dtF1)

    baggingF1 >= dtF1 should be(true)
  }


}


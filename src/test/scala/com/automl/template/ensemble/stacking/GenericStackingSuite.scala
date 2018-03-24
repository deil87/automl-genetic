package com.automl.template.ensemble.stacking

import com.automl.helper.TemplateTreeHelper
import com.automl.spark.SparkSessionProvider
import com.automl.template._
import com.automl.template.ensemble.bagging.Bagging
import com.automl.template.simple._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{GBTRegressor, LinearRegression}
import org.apache.spark.sql.DataFrame
import org.scalatest.{FunSuite, Matchers}
import utils.SparkMLUtils


class GenericStackingSuite extends FunSuite with Matchers with SparkSessionProvider{

  import utils.SparkMLUtils._

  lazy val airlineDF = SparkMLUtils.loadResourceDF("/airline2008-100k_rows.csv")
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
    .limit(3000)
    .applyTransformation(featuresAssembler)
    .withColumnRenamed("DepDelay", "label")
    .toDouble("label")
    .filterOutNull("label")
    .withColumn("uniqueIdColumn", monotonically_increasing_id)
    .printSchema_AndContinue
    .showN_AndContinue(100)
    .cache()


  test("We can train on this dataset with base members") {

    val linearRegression = new LinearRegression()

    val Array(trainingSplit, testSplit) = prepairedAirlineDF.randomSplit(Array(0.8, 0.2))
    noException shouldBe thrownBy(  linearRegression.fit(trainingSplit))

  }

  test("Generic stacking member should calculate fitness over complex tree of members and with better performance than each particular member") {

    val models = Seq(
      LeafTemplate(new LinearRegressionModel()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(RandomForest()),
//      LeafTemplate(Bayesian()), // TODO why Bayesian model is so bad?
      NodeTemplate(Bagging(), Seq(
        LeafTemplate(new LinearRegressionModel()),
        LeafTemplate(GradientBoosting()),
        LeafTemplate(DecisionTree())
      ))
    )

    val Array(trainingSplit, testSplit) = prepairedAirlineDF.randomSplit(Array(0.8, 0.2))

    val genericStacking = GenericStacking(metaLearner = new GBTRegressor())

    val fitnessResult = genericStacking.ensemblingFitnessError(trainingSplit, testSplit, models)

    val rmseFromLR = computeWithLR(trainingSplit, testSplit)
    println(s"RMSE computed for Linear regression model $rmseFromLR")

    fitnessResult.fitnessError should be <= rmseFromLR

    val rmseFromGB = GradientBoosting().fitnessError(trainingSplit, testSplit).fitnessError
    println(s"RMSE computed for GradientBoosting model $rmseFromGB")

    fitnessResult.fitnessError should be <= rmseFromGB

  }

  def computeWithLR(trainDF: DataFrame, testDF: DataFrame): Double = {

    val linearRegression = new LinearRegression()

    val model = linearRegression.fit(trainDF)

    val predictions = model.transform(testDF)

    predictions.cache()

    val evaluator = new RegressionEvaluator()

    evaluator.evaluate(predictions)
  }


}


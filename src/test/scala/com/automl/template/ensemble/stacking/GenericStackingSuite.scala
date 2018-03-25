package com.automl.template.ensemble.stacking

import com.automl.spark.SparkSessionProvider
import com.automl.template._
import com.automl.template.ensemble.bagging.Bagging
import com.automl.template.simple._
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.regression.{GBTRegressor, LinearRegression}
import org.apache.spark.storage.StorageLevel
import org.scalatest.{FunSuite, Matchers}
import utils.{BenchmarkHelper, SparkMLUtils}


class GenericStackingSuite extends FunSuite with Matchers with SparkSessionProvider{

  import utils.SparkMLUtils._

  lazy val airlineDF = SparkMLUtils.loadResourceDF("/airline_sampled_100k.csv")
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

  val scaler = new StandardScaler()
    .setInputCol("features")
    .setOutputCol("scaledFeatures")
    .setWithStd(false)
    .setWithMean(true)

  import org.apache.spark.sql.functions.monotonically_increasing_id

  val prepairedAirlineDF = airlineDF
    .limit(3000)
    .applyTransformation(featuresAssembler)
    .applyTransformation(scaler)
    .showN_AndContinue(10)
    .withColumnReplace("features", "scaledFeatures")
    .showN_AndContinue(10)
    .withColumnRenamed("DepDelay", "label")
    .toDouble("label")
    .filterOutNull("label")
    .withColumn("uniqueIdColumn", monotonically_increasing_id)
    .printSchema_AndContinue
    .showN_AndContinue(100)

  val Array(trainingSplit, testSplit) = prepairedAirlineDF.randomSplit(Array(0.8, 0.2))

  trainingSplit.persist(StorageLevel.MEMORY_ONLY)
  testSplit.persist(StorageLevel.MEMORY_ONLY)

  test("We can train on this dataset with base members") {

    val linearRegression = new LinearRegression()

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

    val genericStacking = GenericStacking(metaLearner = new GBTRegressor())

    val fitnessResult = genericStacking.ensemblingFitnessError(trainingSplit, testSplit, models)

    val rmseFromLR = LinearRegressionModel().fitnessError(trainingSplit, testSplit).fitnessError
    println(s"RMSE computed for Linear regression model $rmseFromLR")

    fitnessResult.fitnessError should be <= rmseFromLR

    val rmseFromGB = GradientBoosting().fitnessError(trainingSplit, testSplit).fitnessError
    println(s"RMSE computed for GradientBoosting model $rmseFromGB")

    fitnessResult.fitnessError should be <= rmseFromGB

  }

  test("Generic stacking member should calculate fitness over tree of height 3 and with better performance than each particular member") {

    val models = Seq(
      LeafTemplate(DecisionTree()),
      NodeTemplate(GenericStacking(), Seq(
        NodeTemplate(Bagging(), Seq(
          LeafTemplate(LinearRegressionModel())
        )),
        NodeTemplate(Bagging(), Seq(
          LeafTemplate(LinearRegressionModel())
        ))
      ))
    )

    BenchmarkHelper.time("stacking") {

      val genericStacking = GenericStacking(metaLearner = new GBTRegressor())

      val fitnessResult = genericStacking.ensemblingFitnessError(trainingSplit, testSplit, models)

      val rmseFromLR = LinearRegressionModel().fitnessError(trainingSplit, testSplit).fitnessError
      println(s"RMSE computed for Linear regression model $rmseFromLR")

      fitnessResult.fitnessError should be <= rmseFromLR

      val rmseFromGB = GradientBoosting().fitnessError(trainingSplit, testSplit).fitnessError
      println(s"RMSE computed for GradientBoosting model $rmseFromGB")

      fitnessResult.fitnessError should be <= rmseFromGB

    }
  }
}


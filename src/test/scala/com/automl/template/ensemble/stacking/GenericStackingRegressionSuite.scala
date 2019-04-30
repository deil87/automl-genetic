package com.automl.template.ensemble.stacking

import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.problemtype.ProblemType
import com.automl.spark.SparkSessionProvider
import com.automl.template._
import com.automl.template.simple._
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.regression.{GBTRegressor, LinearRegression}
import org.apache.spark.storage.StorageLevel
import org.scalatest.{FunSuite, Matchers}
import utils.{BenchmarkHelper, SparkMLUtils}

import scala.util.Random


class GenericStackingRegressionSuite extends FunSuite with Matchers with SparkSessionProvider{

  import utils.SparkMLUtils._

  lazy val airlineDF = SparkMLUtils.loadParquet("src/test/resources/airline_allcolumns_sampled_100k_parquet")
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
    .setWithStd(true)
    .setWithMean(false)

  import org.apache.spark.sql.functions.monotonically_increasing_id

  val prepairedAirlineDF = airlineDF
    .limit(3000)
    .applyTransformation(featuresAssembler)
    .applyTransformation(scaler)
//    .showN_AndContinue(10)
    .withColumnReplace("features", "scaledFeatures")
//    .showN_AndContinue(10)
    .withColumnRenamed("DepDelay", "label")
    .toDouble("label")
    .filterOutNull("label")
    .withColumn("uniqueIdColumn", monotonically_increasing_id)
//    .printSchema_AndContinue
    .showN_AndContinue(30)

  val Array(trainingSplit, testSplit) = prepairedAirlineDF.randomSplit(Array(0.8, 0.2))

  trainingSplit.persist(StorageLevel.MEMORY_ONLY)
  testSplit.persist(StorageLevel.MEMORY_ONLY)

  test("We can train on this dataset with base members") {

    val linearRegression = new LinearRegression()

    noException shouldBe thrownBy(  linearRegression.fit(trainingSplit))

  }

  ignore("Generic stacking member should calculate fitness over complex tree of members and with better performance than each particular member") {

    val models = Seq(
      LeafTemplate(new LinearRegressionModel()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(RandomForest()),
//      LeafTemplate(Bayesian()), // TODO why Bayesian model is so bad?
      NodeTemplate(SparkGenericBagging(), Seq(
        LeafTemplate(new LinearRegressionModel()),
        LeafTemplate(GradientBoosting()),
        LeafTemplate(DecisionTree())
      ))
    )

    val genericStacking = GenericStacking(unusedMetaLearner = new GBTRegressor())

    val problemType = ProblemType.RegressionProblem

    val seed = new Random().nextLong()

    val fitnessResult = genericStacking.ensemblingFitnessError(trainingSplit, testSplit, models, problemType, hyperParamsField = Some(HyperParametersField.default), seed = seed)

    val rmseFromLR = LinearRegressionModel().fitnessError(trainingSplit, testSplit).getCorrespondingMetric
    println(s"RMSE computed for Linear regression model $rmseFromLR")

    fitnessResult.getCorrespondingMetric should be <= rmseFromLR

    val rmseFromGB = GradientBoosting().fitnessError(trainingSplit, testSplit).getCorrespondingMetric
    println(s"RMSE computed for GradientBoosting model $rmseFromGB")

    fitnessResult.getCorrespondingMetric should be <= rmseFromGB

  }

  test("Generic stacking member should calculate fitness over tree of height 3 and with better performance than each particular member") {

    val models = Seq(
      LeafTemplate(DecisionTree()),
      NodeTemplate(GenericStacking(), Seq(
        NodeTemplate(SparkGenericBagging(), Seq(
          LeafTemplate(LinearRegressionModel())
        )),
        NodeTemplate(SparkGenericBagging(), Seq(
          LeafTemplate(LinearRegressionModel())
        ))
      ))
    )

    BenchmarkHelper.time("GenericStackingSuite.scala ") {

      val genericStacking = GenericStacking(unusedMetaLearner = new GBTRegressor())

      val problemType = ProblemType.RegressionProblem

      val seed = new Random().nextLong()

      val fitnessResult = genericStacking.ensemblingFitnessError(trainingSplit, testSplit, models, problemType, hyperParamsField = Some(HyperParametersField.default), seed = seed)

      val rmseFromLR = LinearRegressionModel().fitnessError(trainingSplit, testSplit, problemType).getCorrespondingMetric
      println(s"RMSE computed for Linear regression model $rmseFromLR")

      fitnessResult.getCorrespondingMetric should be <= rmseFromLR

      val rmseFromGB = GradientBoosting().fitnessError(trainingSplit, testSplit, problemType).getCorrespondingMetric
      println(s"RMSE computed for GradientBoosting model $rmseFromGB")

      fitnessResult.getCorrespondingMetric should be <= rmseFromGB

    }(0)
  }
}


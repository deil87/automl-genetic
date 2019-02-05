package com.automl.template.simple

import com.automl.TPopulation
import com.automl.problemtype.ProblemType
import com.automl.spark.SparkSessionProvider
import com.automl.template.LeafTemplate
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.functions.rand
import org.scalatest.{FunSuite, Matchers}
import utils.SparkMLUtils

class BayesianTest extends FunSuite with SparkSessionProvider with Matchers{

  import utils.SparkMLUtils._
  import ss.implicits._

  val seed = Seq(
    LeafTemplate(LogisticRegressionModel()),
    LeafTemplate(Bayesian()),
    //      LeafTemplate(GradientBoosting()), //TODO multiclass classification case is not supported
    //        LeafTemplate(NeuralNetwork(Array(5,10,5))), // TODO need to implement detection of features number and number of classes
    LeafTemplate(DecisionTree())
  )

  val seedPopulation = new TPopulation(seed)

  val population = TPopulation.fromSeedPopulation(seedPopulation).withSize(10).build

  val glassDF = SparkMLUtils.loadResourceDF("/dataset/glass/glass.csv")

  val features = Array("RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe")

  val featuresColName: String = "features"

  def featuresAssembler = {
    new VectorAssembler()
      .setInputCols(features)
      .setOutputCol(featuresColName)
  }

  val scaler = new StandardScaler()
    .setInputCol("features")
    .setOutputCol("scaledFeatures")
    .setWithStd(true)
    .setWithMean(false)

  // see https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3741049972324885/3783546674231736/4413065072037724/latest.html
  val preparedGlassDF = glassDF
    .orderBy(rand())  // Shuffling
    .applyTransformation(featuresAssembler)
    .applyTransformation(scaler)
    .drop("features")
    .withColumnRenamed("scaledFeatures", "features")
    .toLong("Id")
    .withColumnRenamed("Id", "uniqueIdColumn")
    .withColumnRenamed("TypeOfGlass", "indexedLabel")
    .withColumnReplace("indexedLabel", $"indexedLabel" - 1.0 ) //TODO need to do this as Bayesian expects zero based
    .toDouble("indexedLabel")
    .showN_AndContinue(10)
    .printSchema_AndContinue
    .cache()

  val Array(trainDF, testDF) = preparedGlassDF.randomSplit(Array(0.8, 0.2))
  trainDF.cache()
  testDF.cache()

  val problemType = ProblemType.MultiClassClassificationProblem

  test("testFitnessError") {
    val bayesianF1 = Bayesian().fitnessError(trainDF, testDF, problemType).getCorrespondingMetric
    println(s"F1 computed for Bayesian model $bayesianF1")

    bayesianF1 shouldBe 0.9 +- 0.1
  }

}

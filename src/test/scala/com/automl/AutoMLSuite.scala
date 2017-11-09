package com.automl

import com.automl.template._
import com.automl.template.simple.{Bayesian, DecisionTree, SimpleModelMember}
import ml.dmlc.xgboost4j.scala.spark.XGBoostEstimator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.SparkSession
import org.scalatest.{FunSuite, Matchers}
import utils.SparkMLUtils


class AutoMLSuite extends FunSuite with Matchers {

  implicit val ss: SparkSession = SparkSession.builder()
    .master("local[*]")
    .appName("Kaggle")
    .config("spark.cassandra.connection.host", "127.0.0.1")
    .config("spark.driver.memory", "6g")
    .config("spark.executor.memory", "6g")
    .getOrCreate()


  ss.sparkContext.setLogLevel("INFO")

  import utils.SparkMLUtils._


  test("AutoML should mutate templateTree from base model to complex algorithm") {

    val population: Seq[LeafTemplate[SimpleModelMember]] = Seq(LeafTemplate(Bayesian()), LeafTemplate(DecisionTree()))

    val autoMl = new AutoML(null, 500000, useMetaDB = false)

    val mutated = autoMl.applyMutation(population)
    val mutated2 = autoMl.applyMutation(mutated)
    val mutated3 = autoMl.applyMutation(mutated2)

    val heightsOfIndividuals = mutated.map(_.height)

    //TODO make mutation happens every time
    mutated shouldNot be(population)
    mutated2 shouldNot be(mutated)
    mutated3 shouldNot be(mutated2)
  }



  test("AutoML should run UCI airline dataset and compute performance metrics for base models") {


    val airlineDF = SparkMLUtils.loadResourceDF("/airline2008.csv")
      .select("DayOfWeek", "Distance", "DepTime", "CRSDepTime", "DepDelay")

    val res = airlineDF.show(50)

    val features = Array("Distance", "DayOfWeek")
    val oheFeatures = Array.empty

    val combinedFeatures = features /*++ oheFeatures*/

    val featuresColName: String = "features"

    def featuresAssembler = {
      new VectorAssembler()
        .setInputCols(combinedFeatures)
        .setOutputCol(featuresColName)
    }

    val prepairedAirlineDF = airlineDF
      .applyTransformation(featuresAssembler)
      .withColumnRenamed("DepDelay", "label")
      .toDouble("label")
      .filterOutNull("label")
      .showN_AndContinue(100)
      .cache()

    lazy val Array(trainingSplit, testSplit) = prepairedAirlineDF.randomSplit(Array(0.80, 0.20), 11L)

    val xGBoostEstimator = new XGBoostEstimator("DepDelayPredictor")
    val pipeline = new Pipeline().setStages(Array(xGBoostEstimator))


    val paramsMap: Array[ParamMap] = new ParamGridBuilder()
      .addGrid(xGBoostEstimator.objective, Array("reg:linear"))
      .addGrid(xGBoostEstimator.evalMetric, Array("rmse"))
      .build()

    val model = xGBoostEstimator.fit(trainingSplit)
    val predictionsForTraining = model.transform(trainingSplit).cache()

    val evaluator = new RegressionEvaluator()

    val rmse = evaluator.evaluate(predictionsForTraining)

    println("Training error (RMSE) =" + rmse)
    assert(rmse > 0)

  }

}


package com.automl

import com.automl.helper.{PopulationHelper, TemplateTreeHelper}
import com.automl.spark.SparkSessionProvider
import com.automl.template._
import com.automl.template.simple.{Bayesian, DecisionTree, LinearRegressionModel, SimpleModelMember}
import ml.dmlc.xgboost4j.scala.spark.XGBoostEstimator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.SparkSession
import org.scalatest.{FunSuite, Matchers}
import utils.SparkMLUtils


class AutoMLSuite extends FunSuite with Matchers with SparkSessionProvider{

//  ss.sparkContext.setLogLevel("INFO")

  import utils.SparkMLUtils._


  test("AutoML should mutate templateTree from base model to complex algorithm") {

    val seed: Seq[LeafTemplate[SimpleModelMember]] = Seq(LeafTemplate(Bayesian()), LeafTemplate(DecisionTree()))

    val seedPopulation = new Population(seed)

    val population = Population.fromSeedPopulation(seedPopulation).withSize(10).build

    val autoMl = new AutoML(null, 50000, useMetaDB = false, initialPopulationSize = Some(10))

    PopulationHelper.print(population)

    val mutated = autoMl.applyMutation(population)

    PopulationHelper.print(mutated)

    val mutated2 = autoMl.applyMutation(mutated)
    PopulationHelper.print(mutated2)

    val mutated3 = autoMl.applyMutation(mutated2)
    PopulationHelper.print(mutated3)

    //TODO make mutation happens every time
    mutated shouldNot be(population)
    mutated2 shouldNot be(mutated)
    mutated3 shouldNot be(mutated2)
  }

  test("AutoML should find best template with most optimal fitness value") {

    val seed: Seq[LeafTemplate[SimpleModelMember]] = Seq(LeafTemplate(LinearRegressionModel()), LeafTemplate(DecisionTree()))

    val seedPopulation = new Population(seed)

    val population = Population.fromSeedPopulation(seedPopulation).withSize(10).build

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
      .limit(3000)
      .applyTransformation(featuresAssembler)
      .withColumnRenamed("DepDelay", "label")
      .toDouble("label")
      .filterOutNull("label")
      .withColumn("uniqueIdColumn", monotonically_increasing_id)
      .showN_AndContinue(100)
      .cache()

    val Array(trainingSplit, testSplit) = prepairedAirlineDF.randomSplit(Array(0.8, 0.2))

    trainingSplit.cache()

    val autoMl = new AutoML(trainingSplit, 50000, useMetaDB = false, initialPopulationSize = Some(10), seedPopulation = seedPopulation)

    autoMl.run()
  }



  ignore("AutoML should run UCI airline dataset and compute performance metrics for base models") {


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


package com.automl

import com.automl.evolution.dimension.{TemplateEvolutionDimension, TemplateHyperParametersEvolutionDimension}
import com.automl.spark.SparkSessionProvider
import com.automl.template._
import com.automl.template.simple._
import kamon.Kamon
import org.apache.spark.ml.feature.VectorAssembler
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpec}
import utils.SparkMLUtils


class AutoMLMainSuite extends WordSpec with Matchers with BeforeAndAfterAll with SparkSessionProvider {

  ss.sparkContext.setLogLevel("ERROR")

  import utils.SparkMLUtils._

  "AutoML" should {

    "find best template" in {

      val seed: Seq[LeafTemplate[SimpleModelMember]] = Seq(
        LeafTemplate(LinearRegressionModel()),
        LeafTemplate(Bayesian()),
        LeafTemplate(GradientBoosting()),
//        LeafTemplate(NeuralNetwork(Array(5,10,5))), // TODO need to implement detection of features number and number of classes
        LeafTemplate(DecisionTree())
      )

      val seedPopulation = new Population(seed)

      val population = Population.fromSeedPopulation(seedPopulation).withSize(10).build

      val airlineDF = SparkMLUtils.loadParquet("src/test/resources/airline_allcolumns_sampled_100k_parquet")
        .select("DayOfWeek", "Distance", "DepTime", "CRSDepTime", "DepDelay")

      val features = Array("Distance", "DayOfWeek")

      val combinedFeatures = features

      val featuresColName: String = "features"

      def featuresAssembler = {
        new VectorAssembler()
          .setInputCols(combinedFeatures)
          .setOutputCol(featuresColName)
      }
      import org.apache.spark.sql.functions.monotonically_increasing_id

      val preparedAirlineDF = airlineDF
        .limit(30000)
        .applyTransformation(featuresAssembler)
        .withColumnRenamed("DepDelay", "label")
        .toDouble("label")
        .filterOutNull("label")
        .withColumn("uniqueIdColumn", monotonically_increasing_id)
        .showN_AndContinue(10)
        .cache()

      val Array(trainingSplit, testSplit) = preparedAirlineDF.randomSplit(Array(0.8, 0.2)) // TODO testSplit is never used

      trainingSplit.cache()

      val autoMl = new AutoML(
        data = trainingSplit,
        maxTime = 300000,
        useMetaDB = false,
        initialPopulationSize = Some(7),
        seedPopulation = seedPopulation,
        maxGenerations = 5)

      val templateEvDim = new TemplateEvolutionDimension
      val hyperParamsEvDim = new TemplateHyperParametersEvolutionDimension

      autoMl.runEvolution(templateEvDim, hyperParamsEvDim)

    }
  }

  override protected def afterAll(): Unit = {
//    ss.stop()
  }
}


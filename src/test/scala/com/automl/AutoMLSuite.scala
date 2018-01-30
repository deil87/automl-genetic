package com.automl

import com.automl.helper.{FitnessResult, PopulationHelper}
import com.automl.spark.SparkSessionProvider
import com.automl.template._
import com.automl.template.ensemble.bagging.Bagging
import com.automl.template.simple._
import kamon.Kamon
import org.apache.spark.ml.feature.VectorAssembler
import org.scalatest.{Matchers, WordSpec}
import org.slf4j.LoggerFactory
import utils.SparkMLUtils

import scala.util.Random


class AutoMLSuite extends WordSpec with Matchers with SparkSessionProvider {

  ss.sparkContext.setLogLevel("ERROR")


  import kamon.prometheus.PrometheusReporter
  import utils.SparkMLUtils._
  Kamon.addReporter(new PrometheusReporter())

  trait Fixture {
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
      .limit(5000)
      .applyTransformation(featuresAssembler)
      .withColumnRenamed("DepDelay", "label")
      .toDouble("label")
      .filterOutNull("label")
      .withColumn("uniqueIdColumn", monotonically_increasing_id)
      .showN_AndContinue(10)
      .cache()

    val Array(trainingSplit, testSplit) = prepairedAirlineDF.randomSplit(Array(0.8, 0.2))

    trainingSplit.cache()

  }

  "AutoML" should {

    "mutate templateTree from base model to complex algorithm" ignore {

      val seed: Seq[LeafTemplate[SimpleModelMember]] = Seq(
        LeafTemplate(Bayesian()),
        LeafTemplate(LinearRegressionModel()),
        LeafTemplate(DecisionTree())
      )

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

    "find best template with most optimal fitness value" in {

      val seed: Seq[LeafTemplate[SimpleModelMember]] = Seq(
        LeafTemplate(LinearRegressionModel()),
        LeafTemplate(Bayesian()),
        LeafTemplate(DecisionTree())
      )

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
//        .limit(5000)
        .applyTransformation(featuresAssembler)
        .withColumnRenamed("DepDelay", "label")
        .toDouble("label")
        .filterOutNull("label")
        .withColumn("uniqueIdColumn", monotonically_increasing_id)
        .showN_AndContinue(10)
        .cache()

      val Array(trainingSplit, testSplit) = prepairedAirlineDF.randomSplit(Array(0.8, 0.2))

      trainingSplit.cache()

      val autoMl = new AutoML(trainingSplit, 30000, useMetaDB = false, initialPopulationSize = Some(7), seedPopulation = seedPopulation, maxGenerations = 5)

      autoMl.run()

    }

    /* test("we take values from cache correctly even for complex templates") {

     }
   */
    "use different cache values for same template but different dataframe" in new Fixture {
      val seed: Seq[LeafTemplate[SimpleModelMember]] = Seq(
        LeafTemplate(Bayesian()),
        LeafTemplate(LinearRegressionModel()),
        LeafTemplate(DecisionTree())
      )

      val population = new Population(seed)

      val autoMl = new AutoML(trainingSplit, 300000, useMetaDB = false, initialPopulationSize = Some(10), seedPopulation = population, maxGenerations = 5)

      val ds2 = trainingSplit.limit(20)
      val ds3 = trainingSplit.limit(300)
      autoMl.calculateFitnessResults(population, ds2)
      autoMl.calculateFitnessResults(population, ds3)

      autoMl.individualsCache.size should be (6)  // (numbers of templates in population) * (# of different sizes of training datasets)
    }

    "caching is working within ensemble nodes" in new Fixture{

      val autoMl = new AutoML(null, 300000, useMetaDB = false, initialPopulationSize = Some(10), seedPopulation = null, maxGenerations = 5)

      val template = Seq(
        NodeTemplate(Bagging(), Seq(
          LeafTemplate(LinearRegressionModel()),
          NodeTemplate(Bagging(), Seq(
            LeafTemplate(LinearRegressionModel()),
            LeafTemplate(LinearRegressionModel())
          ))
        )
        )
      )

      val testPopulation = new Population(template)

      val ds2 = trainingSplit.limit(20)
      autoMl.calculateFitnessResults(testPopulation, ds2)
      true shouldBe true
    }


    "parentSelectionByFitnessRank" ignore {
      val autoMl = new AutoML(null, 50000, useMetaDB = false, initialPopulationSize = Some(10))
      val individuals: Seq[LeafTemplate[SimpleModelMember]] = Seq(
        LeafTemplate(LinearRegressionModel()),
        LeafTemplate(DecisionTree())
      )

      val individualsSpanned = Population.fromSeedPopulation(new Population(individuals)).withSize(1000).build.individuals

      val selectedParents = autoMl.parentSelectionByFitnessRank(0.5, individualsSpanned.zipWithIndex.map { case (inds, idx) => IndividualAlgorithmData(idx.toString, inds, null, FitnessResult(Random.nextDouble(), null)) })

      import breeze.plot._

      val f = Figure()
      val p2 = f.subplot(0)
      p2 += hist(selectedParents.map(_.rank), 100)
      p2.title = "A normal distribution"
      f.saveas("subplots.png")
      Thread.sleep(50000)

      PopulationHelper.print(new Population(selectedParents.map(_.template)))

      selectedParents.length shouldBe 50

    }

    //  ignore("AutoML should run UCI airline dataset and compute performance metrics for base models") {
    //
    //
    //    val airlineDF = SparkMLUtils.loadResourceDF("/airline2008.csv")
    //      .select("DayOfWeek", "Distance", "DepTime", "CRSDepTime", "DepDelay")
    //
    //    val res = airlineDF.show(50)
    //
    //    val features = Array("Distance", "DayOfWeek")
    //    val oheFeatures = Array.empty
    //
    //    val combinedFeatures = features /*++ oheFeatures*/
    //
    //    val featuresColName: String = "features"
    //
    //    def featuresAssembler = {
    //      new VectorAssembler()
    //        .setInputCols(combinedFeatures)
    //        .setOutputCol(featuresColName)
    //    }
    //
    //    val prepairedAirlineDF = airlineDF
    //      .applyTransformation(featuresAssembler)
    //      .withColumnRenamed("DepDelay", "label")
    //      .toDouble("label")
    //      .filterOutNull("label")
    //      .showN_AndContinue(100)
    //      .cache()
    //
    //    lazy val Array(trainingSplit, testSplit) = prepairedAirlineDF.randomSplit(Array(0.80, 0.20), 11L)
    //
    //    val xGBoostEstimator = new XGBoostEstimator("DepDelayPredictor")
    //    val pipeline = new Pipeline().setStages(Array(xGBoostEstimator))
    //
    //
    //    val paramsMap: Array[ParamMap] = new ParamGridBuilder()
    //      .addGrid(xGBoostEstimator.objective, Array("reg:linear"))
    //      .addGrid(xGBoostEstimator.evalMetric, Array("rmse"))
    //      .build()
    //
    //    val model = xGBoostEstimator.fit(trainingSplit)
    //    val predictionsForTraining = model.transform(trainingSplit).cache()
    //
    //    val evaluator = new RegressionEvaluator()
    //
    //    val rmse = evaluator.evaluate(predictionsForTraining)
    //
    //    println("Training error (RMSE) =" + rmse)
    //    assert(rmse > 0)
    //
    //  }

  }
}


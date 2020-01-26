package com.automl

import akka.actor.ActorSystem
import com.automl.dataset.Datasets
import com.automl.population.{GenericPopulationBuilder, TPopulation}
import com.automl.spark.SparkSessionProvider
import com.automl.template._
import com.automl.template.simple._
import org.apache.spark.ml.feature.VectorAssembler
import org.scalatest.{Matchers, WordSpec}
import utils.SparkMLUtils


class AutoMLSuite extends WordSpec with Matchers with SparkSessionProvider {

  import utils.SparkMLUtils._

  implicit val system = ActorSystem("AutoMLSuite-system")
  implicit val logPaddingSize: Int = 0

  "AutoML" should {

    // ignored as it is more like a Benchmark not a unit test
    "find best template with most optimal fitness value" ignore {

      val seed: Seq[LeafTemplate[SimpleModelMember]] = Seq(
        LeafTemplate(LinearRegressionModel()),
        LeafTemplate(Bayesian()),
        LeafTemplate(GradientBoosting()),
        LeafTemplate(DecisionTree())
      )

      val seedPopulation = new TPopulation(seed)

      val population = GenericPopulationBuilder.fromSeedPopulation(seedPopulation).withSize(10).build

      val airlineDF = SparkMLUtils.loadParquet("src/test/resources/airline_allcolumns_sampled_100k_parquet")
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
        .limit(15000)
        .applyTransformation(featuresAssembler)
        .withColumnRenamed("DepDelay", "label")
        .toDouble("label")
        .filterOutNull("label")
        .withColumn("uniqueIdColumn", monotonically_increasing_id)
//        .showN_AndContinue(10)
        .cache()

      val Array(trainingSplit, testSplit) = prepairedAirlineDF.randomSplit(Array(0.8, 0.2))

      trainingSplit.cache()

      val autoMl = new AutoML(trainingSplit, maxTime = 300000, useMetaDB = false, initialPopulationSize = Some(7), seedPopulation = Some(seedPopulation), maxGenerations = 5)

      autoMl.runEvolution(system)

    }

    "best model was selected with accordance to chosen metric ( f1 is theBiggerTheBetter metric) " in {

      ConfigProvider.clearOverride.addOverride(
        """
          |evolution {
          |  hyperParameterDimension {
          |    enabled = false
          |  }
          |  evaluation {
          |    multiclass.metric = "f1"
          |  }
          |}
        """)

      implicit val system = ActorSystem("AutoML-system")

      val individuals = Seq(
        LeafTemplate(LogisticRegressionModel()),
        LeafTemplate(Bayesian()),
        LeafTemplate(DecisionTree())
      )

      val seedPopulation = new TPopulation(individuals)

      val seed = 1234
      val preparedGlassDF = Datasets.getGlassDataFrame(seed)

      //Note we are passing whole dataset and inside it is being splitted as train/test. Maybe it is a good idea to hold test split for a final examination.
      val autoMl = new AutoML(
        data = preparedGlassDF,
        responseColumn = "indexedLabel",
        maxTime = 15000,
        useMetaDB = false,
        initialPopulationSize = Some(3),
        seedPopulation = Some(seedPopulation),
        maxEvolutions = 5,
        isBigSizeThreshold = 100,
        initialSampleSize = 50)

      val hallOfFame = autoMl.runEvolution

      assume(hallOfFame.size >= 2)
      val best = hallOfFame.dequeue()
      val secondButTheBest = hallOfFame.dequeue()
      best.result.getCorrespondingMetric should be >= secondButTheBest.result.getCorrespondingMetric

    }

    "best model was selected with accordance to chosen metric ( logloss is theSmallerTheBetter metric) " in {

      val metric = "logloss"

      ConfigProvider.clearOverride.addOverride(
        s"""
          |evolution {
          |  hyperParameterDimension {
          |    enabled = false
          |  }
          |  evaluation {
          |    multiclass.metric = "$metric"
          |  }
          |}
        """)

      implicit val system = ActorSystem("AutoML-system")

      val individuals = Seq(
        LeafTemplate(LogisticRegressionModel()),
        LeafTemplate(Bayesian()),
        LeafTemplate(RandomForest()),
        LeafTemplate(DecisionTree())
//        LeafTemplate(SVMModel()) // TODO AG-184
      )

      val seedPopulation = new TPopulation(individuals)

      val seed = 1234
      val preparedGlassDF = Datasets.getGlassDataFrame(seed)

      //Note we are passing whole dataset and inside it is being splitted as train/test. Maybe it is a good idea to hold test split for a final examination.
      val autoMl = new AutoML(
        data = preparedGlassDF,
        responseColumn = "indexedLabel",
        maxTime = 120000,
        useMetaDB = false,
        initialPopulationSize = Some(3),
        seedPopulation = Some(seedPopulation),
        maxEvolutions = 5,
        isBigSizeThreshold = 215,
        initialSampleSize = 50)

      val hallOfFame = autoMl.runEvolution

      assume(hallOfFame.size >= 2)

      val best = hallOfFame.dequeue()
      val secondButTheBest = hallOfFame.dequeue()
      best.result.getCorrespondingMetric should be <= secondButTheBest.result.getCorrespondingMetric
    }

  }
}


package com.automl.benchmark.glass

import akka.actor.ActorSystem
import com.automl.dataset.Datasets
import com.automl.population.{GenericPopulationBuilder, TPopulation}
import com.automl.spark.SparkSessionProvider
import com.automl.template.LeafTemplate
import com.automl.template.simple._
import com.automl.{AutoML, ConfigProvider}
import com.typesafe.config.{Config, ConfigFactory, ConfigRenderOptions}

class GlassOnlyDecisionTreesEnsembleRuns(implicit as: ActorSystem) extends SparkSessionProvider{

  def depthOneRun() = {

    val testOverride: Config = ConfigFactory.parseString(
      """
        |evolution {
        |  templateDimension {
        |    populationSize = 5
        |    poolOfSimpleModels = ["decision_tree"]
        |    poolOfEnsemblingModels = ["bagging", "stacking"]
        |    maxEnsembleDepth = 1
        |  }
        |  hyperParameterDimension {
        |    enabled = false
        |  }
        |}
      """.stripMargin)
    ConfigProvider.addOverride(testOverride)
    println(ConfigProvider.config.root().render(ConfigRenderOptions.concise()))

    val individuals = Seq(
      LeafTemplate(DecisionTree())
    )

    val seedPopulation = new TPopulation(individuals)

    val population = GenericPopulationBuilder.fromSeedPopulation(seedPopulation).withSize(7).build
    val seed = 1234
    val preparedGlassDF = Datasets.getGlassDataFrame(seed)

    //Note we are passing whole dataset and inside it is being splitted as train/test. Maybe it is a good idea to hold test split for a final examination.
    val autoMl = new AutoML(
      data = preparedGlassDF,
      responseColumn = "indexedLabel",
      maxTime = 6 * 60000,
      useMetaDB = false,
      initialPopulationSize = Some(7),
      seedPopulation = Some(seedPopulation),
      maxGenerations = 5)

    autoMl.runEvolution(as)
  }


  def depthTwoRun() = {

    val testOverride: Config = ConfigFactory.parseString(
      """
        |evolution {
        |  templateDimension {
        |    populationSize = 7
        |    poolOfSimpleModels = ["decision_tree"]
        |    poolOfEnsemblingModels = ["bagging", "stacking"]
        |    maxEnsembleDepth = 2
        |  }
        |  hyperParameterDimension {
        |    enabled = false
        |  }
        |}
      """.stripMargin)
    ConfigProvider.addOverride(testOverride)
    println(ConfigProvider.config.root().render(ConfigRenderOptions.concise()))

    val individuals = Seq(
      LeafTemplate(DecisionTree())
    )

    val seedPopulation = new TPopulation(individuals)

    val population = GenericPopulationBuilder.fromSeedPopulation(seedPopulation).withSize(7).build
    val seed = 1234
    val preparedGlassDF = Datasets.getGlassDataFrame(seed)

    //Note we are passing whole dataset and inside it is being splitted as train/test. Maybe it is a good idea to hold test split for a final examination.
    val autoMl = new AutoML(
      data = preparedGlassDF,
      responseColumn = "indexedLabel",
      maxTime = 6 * 60000,
      useMetaDB = false,
      initialPopulationSize = Some(7),
      seedPopulation = Some(seedPopulation),
      maxGenerations = 5)

    autoMl.runEvolution(as)
  }

  def depthThreeRun() = {

    val testOverride: Config = ConfigFactory.parseString(
      """
        |evolution {
        |  templateDimension {
        |    populationSize = 7
        |    poolOfSimpleModels = ["decision_tree"]
        |    poolOfEnsemblingModels = ["bagging", "stacking"]
        |    maxEnsembleDepth = 3
        |  }
        |  hyperParameterDimension {
        |    enabled = false
        |  }
        |}
      """.stripMargin)
    ConfigProvider.addOverride(testOverride)
    println(ConfigProvider.config.root().render(ConfigRenderOptions.concise()))

    val individuals = Seq(
      LeafTemplate(DecisionTree())
    )

    val seedPopulation = new TPopulation(individuals)

    val population = GenericPopulationBuilder.fromSeedPopulation(seedPopulation).withSize(7).build
    val seed = 1234
    val preparedGlassDF = Datasets.getGlassDataFrame(seed)

    //Note we are passing whole dataset and inside it is being splitted as train/test. Maybe it is a good idea to hold test split for a final examination.
    val autoMl = new AutoML(
      data = preparedGlassDF,
      responseColumn = "indexedLabel",
      maxTime = 10 * 60000,
      useMetaDB = false,
      initialPopulationSize = Some(7),
      seedPopulation = Some(seedPopulation),
      maxGenerations = 5)

    autoMl.runEvolution(as)
  }

}

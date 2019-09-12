package com.automl.benchmark.balance_scale

import akka.actor.ActorSystem
import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.dataset.Datasets
import com.automl.evolution.dimension.hparameter._
import com.automl.population.TPopulation
import com.automl.spark.SparkSessionProvider
import com.automl.template.simple._
import com.automl.template.{LeafTemplate, NodeTemplate}
import com.automl.{AutoML, ConfigProvider}
import com.typesafe.config.{Config, ConfigFactory, ConfigRenderOptions}

class BalanceScaleDataSetBenchmark(implicit as: ActorSystem) extends SparkSessionProvider{

  def run() = {

    val testOverride: Config = ConfigFactory.parseString(
      """
        |evolution {
        |  templateDimension {
        |    populationSize = 10
        |    poolOfSimpleModels = ["logistic_regression", "decision_tree", "bayesian", "random_forest"]
        |    poolOfEnsemblingModels = ["bagging", "stacking"]
        |    maxEnsembleDepth = 5
        |  }
        |  hyperParameterDimension {
        |    enabled = false
        |  }
        |  hpGridSearch = false
        |}
      """.stripMargin)
    ConfigProvider.addOverride(testOverride)
    println(ConfigProvider.config.root().render(ConfigRenderOptions.concise()))


    val dt = LeafTemplate(DecisionTree())
    dt.internalHyperParamsMap = Some(HyperParametersField(
      Seq(
        DecisionTreeHPGroup(Seq(MaxDepth(Some(7.0))))
      )
    ))
    val bayesian = LeafTemplate(Bayesian())
    bayesian.internalHyperParamsMap = Some(HyperParametersField(
      Seq(
        BayesianHPGroup(Seq(Smoothing(Some(8.0))))
      )
    ))
    val nt = NodeTemplate(SparkGenericBagging(),
      Seq(
        bayesian,
        dt
      )
    )
    bayesian.parent = Some(nt)
    dt.parent = Some(nt)

    // Rest will be added randomly
    val individuals = Seq(
      nt
    )

    val seedPopulation = new TPopulation(individuals)

    val seed = 1234
    val preparedGlassDF = Datasets.getBalanceDataFrame(seed)

    //Note we are passing whole dataset and inside it is being splitted as train/test. Maybe it is a good idea to hold test split for a final examination.
    val autoMl = new AutoML(
      data = preparedGlassDF,
      responseColumn = "indexedLabel",
      maxTime = 10 * 60000,
      useMetaDB = false,
      initialPopulationSize = Some(10),
      seedPopulation = Some(seedPopulation),
      maxGenerations = 5) // TODO do we need max generations?

    autoMl.runEvolution(as)
  }

}

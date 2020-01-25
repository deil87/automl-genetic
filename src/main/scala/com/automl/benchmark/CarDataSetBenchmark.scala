package com.automl.benchmark

import akka.actor.ActorSystem
import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.dataset.Datasets
import com.automl.evolution.dimension.hparameter._
import com.automl.population.TPopulation
import com.automl.spark.SparkSessionProvider
import com.automl.template.ensemble.stacking.GenericStacking
import com.automl.template.{LeafTemplate, NodeTemplate}
import com.automl.template.simple._
import com.automl.{AutoML, ConfigProvider}
import com.typesafe.config.{Config, ConfigFactory, ConfigRenderOptions}

class CarDataSetBenchmark(implicit as: ActorSystem) extends SparkSessionProvider{

  def run() = {

    ConfigProvider.clearOverride.addOverride(
      """
        |evolution {
        |  templateDimension {
        |    populationSize = 10
        |    poolOfSimpleModels = ["logistic_regression", "decision_tree", "bayesian", "random_forest"]
        |    poolOfEnsemblingModels = ["bagging", "stacking"]
        |    maxEnsembleDepth = 3
        |  }
        |  hyperParameterDimension {
        |    enabled = false
        |  }
        |  hpGridSearch = false
        |}
      """)

    println("GlassDataSetBenchmark.print.config: " + ConfigProvider.config.root().render(ConfigRenderOptions.concise()))


    val individuals = Seq(
      LeafTemplate(LogisticRegressionModel()),
      LeafTemplate(Bayesian()),
      LeafTemplate(RandomForest()),
      NodeTemplate(GenericStacking(), Seq(
        LeafTemplate(LogisticRegressionModel()),
        LeafTemplate(Bayesian()),
        LeafTemplate(DecisionTree()),
        LeafTemplate(RandomForest())
      )),
      /*NodeTemplate(SparkGenericBagging(), Seq( // TODO SparkGenericBagging does not have relevant default HPs
        LeafTemplate(LogisticRegressionModel(LogisticRegressionHPGroup(Seq(RegParamLR(Some(0.2)), ElasticNet(Some(0.2)))))),
        LeafTemplate(Bayesian(BayesianHPGroup(Seq(Smoothing(Some(7)))))),
        LeafTemplate(RandomForest(RandomForestHPGroup(Seq(MaxDepthRF(Some(3.0)))))),
        NodeTemplate(SparkGenericBagging(), Seq(
          LeafTemplate(LogisticRegressionModel(LogisticRegressionHPGroup(Seq(RegParamLR(Some(0.5)), ElasticNet(Some(0.9)))))),
          LeafTemplate(Bayesian(BayesianHPGroup(Seq(Smoothing(Some(8)))))),
          LeafTemplate(RandomForest(RandomForestHPGroup(Seq(MaxDepthRF(Some(5.0)))))),
          LeafTemplate(LogisticRegressionModel(LogisticRegressionHPGroup(Seq(RegParamLR(Some(0.1)), ElasticNet(Some(0.3))))))
        ))
      )),*/
      LeafTemplate(DecisionTree())
    )

    val seedPopulation = new TPopulation(individuals)

    val seed = 1234
    val preparedCarDF = Datasets.getCarDataFrame(seed)
//    preparedCarDF.show(20, false)

    //Note we are passing whole dataset and inside it is being splitted as train/test. Maybe it is a good idea to hold test split for a final examination.
    val autoMl = new AutoML(
      data = preparedCarDF,
      responseColumn = "indexedLabel",
      maxTime = 20 * 60000,
      useMetaDB = false,
      initialPopulationSize = Some(3),
      seedPopulation = Some(seedPopulation),
      maxEvolutions = 3,
      isBigSizeThreshold = 100,
      initialSampleSize = 200)

    autoMl.runEvolution(as)
  }

}

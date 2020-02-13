package com.automl.benchmark

import akka.actor.ActorSystem
import com.automl.dataset.Datasets
import com.automl.population.TPopulation
import com.automl.spark.SparkSessionProvider
import com.automl.template.LeafTemplate
import com.automl.template.simple._
import com.automl.{AutoML, ConfigProvider}
import com.typesafe.config.ConfigRenderOptions

class CoverTypeDataSetBenchmark(implicit as: ActorSystem) extends SparkSessionProvider{

  def run() = {

    ConfigProvider.clearOverride.addOverride(
      """
        |evolution {
        |  templateDimension {
        |    populationSize = 10
        |    poolOfSimpleModels = ["logistic_regression", "decision_tree", "bayesian", "random_forest", "neural_network"]
        |    poolOfEnsemblingModels = ["bagging", "stacking"]
        |    maxEnsembleDepth = 3
        |    globalCVNumFolds = 5
        |  }
        |  hyperParameterDimension {
        |    enabled = false
        |  }
        |  hpGridSearch = false
        |}
      """)

    println("CoverTypeDataSetBenchmark.print.config: " + ConfigProvider.config.root().render(ConfigRenderOptions.concise()))


    val individuals = Seq(
      LeafTemplate(LogisticRegressionModel()),
      LeafTemplate(Bayesian()),
      LeafTemplate(RandomForest()),
//      LeafTemplate(OneVsRestModel()),
      LeafTemplate(NeuralNetwork()),
//      NodeTemplate(GenericStacking(), Seq(
//        LeafTemplate(LogisticRegressionModel()),
//        LeafTemplate(Bayesian()),
//        LeafTemplate(DecisionTree()),
//        LeafTemplate(RandomForest())
//      )),
      LeafTemplate(DecisionTree())
    )

    val seedPopulation = new TPopulation(individuals)

    val seed = 1234
    val preparedDF = Datasets.getCoverTypeDataFrame(seed).limit(10000) // TODO !!!!!!!!!!!!!!!!!!!!!!!!!!! limit
    preparedDF.show(20, false)

    //Note we are passing whole dataset and inside it is being splitted as train/test. Maybe it is a good idea to hold test split for a final examination.
    val autoMl = new AutoML(
      data = preparedDF,
      responseColumn = "indexedLabel",
      maxTime = 30 * 60000,
      useMetaDB = false,
      initialPopulationSize = Some(3),
      seedPopulation = Some(seedPopulation),
      maxEvolutions = 3,
      isBigSizeThreshold = 100,
      initialSampleSize = 500)

    autoMl.runEvolution(as)
  }

}

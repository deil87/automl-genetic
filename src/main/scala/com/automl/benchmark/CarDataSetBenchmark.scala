package com.automl.benchmark

import akka.actor.ActorSystem
import com.automl.dataset.Datasets
import com.automl.population.TPopulation
import com.automl.spark.SparkSessionProvider
import com.automl.template.LeafTemplate
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
        |    poolOfSimpleModels = ["logistic_regression", "decision_tree", "bayesian"]
        |    poolOfEnsemblingModels = ["bagging"]
        |    maxEnsembleDepth = 5
        |  }
        |  hyperParameterDimension {
        |    enabled = false
        |  }
        |  hpGridSearch = false
        |}
      """)

    println("GlassDataSetBenchmark.print.config: " + ConfigProvider.config.root().render(ConfigRenderOptions.concise()))


    val individuals = Seq(
//      LeafTemplate(SVMModel()),
      LeafTemplate(LogisticRegressionModel()),
      LeafTemplate(Bayesian()),
//      NodeTemplate(SparkGenericBagging(),
//        Seq(
//          LeafTemplate(Bayesian()),
//          LeafTemplate(DecisionTree())
//        )
//      ),
//      NodeTemplate(GenericStacking(),
//        Seq(
//          LeafTemplate(Bayesian()),
//          LeafTemplate(DecisionTree())
//        )
//      ),
      //      LeafTemplate(GradientBoosting()), //TODO multiclass classification case is not supported
      //        LeafTemplate(NeuralNetwork(Array(5,10,5))), // TODO need to implement detection of features number and number of classes
      LeafTemplate(DecisionTree())/*,
      nt*/
    )

    val seedPopulation = new TPopulation(individuals)

    val seed = 1234
    val preparedCarDF = Datasets.getCarDataFrame(seed)
    preparedCarDF.show(false)

    //Note we are passing whole dataset and inside it is being splitted as train/test. Maybe it is a good idea to hold test split for a final examination.
    val autoMl = new AutoML(
      data = preparedCarDF,
      responseColumn = "indexedLabel",
      maxTime = 3 * 60000,
      useMetaDB = false,
      initialPopulationSize = Some(3),
      seedPopulation = Some(seedPopulation),
      maxEvolutions = 3,
      isBigSizeThreshold = 100,
      initialSampleSize = 200)

    autoMl.runEvolution(as)
  }

}

package com.automl.benchmark.glass

import akka.actor.ActorSystem
import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.dataset.Datasets
import com.automl.evolution.dimension.hparameter._
import com.automl.spark.SparkSessionProvider
import com.automl.template.ensemble.stacking.GenericStacking
import com.automl.template.{LeafTemplate, NodeTemplate}
import com.automl.template.simple._
import com.automl.{AutoML, ConfigProvider}
import com.automl.population.{GenericPopulationBuilder, TPopulation}
import com.automl.template.simple.SVMModel
import com.typesafe.config.{Config, ConfigFactory, ConfigRenderOptions}

class GlassDataSetBenchmark(implicit as: ActorSystem) extends SparkSessionProvider{

  import utils.SparkMLUtils._

  def run() = {

    val testOverride: Config = ConfigFactory.parseString(
      """
        |evolution {
        |  templateDimension {
        |    populationSize = 6
        |    poolOfSimpleModels = ["logistic_regression", "decision_tree", "bayesian"]
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
    val preparedGlassDF = Datasets.getGlassDataFrame(seed)

    //Note we are passing whole dataset and inside it is being splitted as train/test. Maybe it is a good idea to hold test split for a final examination.
    val autoMl = new AutoML(
      data = preparedGlassDF,
      responseColumn = "indexedLabel",
      maxTime = 10 * 60000,
      useMetaDB = false,
      initialPopulationSize = Some(3),
      seedPopulation = Some(seedPopulation),
      maxEvolutions = 5,
      isBigSizeThreshold = 100,
      initialSampleSize = 50)

    autoMl.runEvolution(as)
  }

}

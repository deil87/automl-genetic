package com.automl.benchmark.glass

import akka.actor.ActorSystem
import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.dataset.Datasets
import com.automl.spark.SparkSessionProvider
import com.automl.template.ensemble.stacking.GenericStacking
import com.automl.template.{LeafTemplate, NodeTemplate}
import com.automl.template.simple._
import com.automl.AutoML
import com.automl.population.{GenericPopulationBuilder, TPopulation}
import com.automl.template.simple.SVMModel

class GlassDataSetBenchmark(implicit as: ActorSystem) extends SparkSessionProvider{

  import utils.SparkMLUtils._

  def run() = {

    val individuals = Seq(
//      LeafTemplate(SVMModel()),
      LeafTemplate(LogisticRegressionModel()),
      LeafTemplate(Bayesian()),
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(Bayesian()),
          LeafTemplate(DecisionTree())
        )
      ),
      NodeTemplate(GenericStacking(),
        Seq(
          LeafTemplate(Bayesian()),
          LeafTemplate(DecisionTree())
        )
      ),
      //      LeafTemplate(GradientBoosting()), //TODO multiclass classification case is not supported
      //        LeafTemplate(NeuralNetwork(Array(5,10,5))), // TODO need to implement detection of features number and number of classes
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
      seedPopulation = Some(population),
      maxGenerations = 5)

    autoMl.runEvolution(as)
  }

}

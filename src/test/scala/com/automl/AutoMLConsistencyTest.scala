package com.automl

import akka.actor.ActorSystem
import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.dataset.Datasets
import com.automl.population.TPopulation
import com.automl.spark.SparkSessionProvider
import com.automl.template._
import com.automl.template.simple._
import com.typesafe.config.{Config, ConfigFactory}
import org.scalatest.{Matchers, WordSpec}
import com.test.TestBase


class AutoMLConsistencyTest extends WordSpec with SparkSessionProvider with TestBase{

  implicit val system = ActorSystem("AutoMLSuite-system")
  implicit val logPaddingSize: Int = 0

  "AutoML" should {

    "should be able to find neighbours in the population. For that we need all test instances to be alignd by uniqueID" in {

      val testOverride: Config = ConfigFactory.parseString(
        """
          |evolution {
          |  templateDimension {
          |    populationSize = 2
          |  }
          |}
        """.stripMargin)
      ConfigProvider.addOverride(testOverride)

      val individuals = Seq(
        LeafTemplate(LogisticRegressionModel()),
        LeafTemplate(Bayesian()),
        NodeTemplate(SparkGenericBagging(),
          Seq(
            LeafTemplate(DecisionTree()),
            LeafTemplate(DecisionTree()),
            LeafTemplate(DecisionTree())
          )
        )
      )

      val seedPopulation = new TPopulation(individuals)
      val seed = 1234
      val preparedGlassDF = Datasets.getGlassDataFrame(seed)

      val autoMl = new AutoML(
        data = preparedGlassDF,
        responseColumn = "indexedLabel",
        maxTime = 5 * 60000,
        useMetaDB = false,
        initialPopulationSize = Some(2),
        seedPopulation = Some(seedPopulation),
        maxGenerations = 5) // Maybe we don't need maxGenerations.

      autoMl.runEvolution(system)

    }

  }
}


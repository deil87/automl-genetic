package com.automl.benchmark.glass.winners

import akka.actor.ActorSystem
import com.automl.ConfigProvider
import com.automl.benchmark.MultiRestartBenchmarker
import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.dataset.Datasets
import com.automl.evolution.dimension.hparameter._
import com.automl.helper.TemplateTreeHelper
import com.automl.problemtype.ProblemType
import com.automl.spark.SparkSessionProvider
import com.automl.template.simple.{Bayesian, DecisionTree}
import com.automl.template.{LeafTemplate, NodeTemplate}
import com.typesafe.config.{Config, ConfigFactory, ConfigRenderOptions}
import org.apache.spark.sql.DataFrame
import org.scalatest.{BeforeAndAfterAll, FunSuite, Matchers}

import scala.util.Random


// These tests do not use AutoML but directly compute performance on a specific ensembles
class GlassMultiRestartWinnersBenchmarks extends FunSuite with Matchers with BeforeAndAfterAll with SparkSessionProvider {

  val testOverride: Config = ConfigFactory.parseString(
    """
      |evolution {
      |  hyperParameterDimension {
      |    enabled = false
      |  }
      |  hpGridSearch = false
      |}
    """.stripMargin)
  ConfigProvider.addOverride(testOverride)
  println(ConfigProvider.config.root().render(ConfigRenderOptions.concise()))

  /**
    * Best template: Bagging ensembling member hps:  hasParent:true doe:0.0
    * ├── Bayesian simpleModel member hps: smoothing:6.0 hasParent:true
    * └── DecisionTree simpleModel member hps: max_depth:2.0 hasParent:true Score: f1:0.6667206629608158
    */
  test("Custom ensemble on Glass dataset performance benchmark") {

    val seed = new Random().nextLong()
    println(s"Seed for current test: $seed")
    val data = Datasets.getGlassDataFrame(seed)
    performTestOn(data, seed)

  }

  private def performTestOn(data: DataFrame, seed: Long) = {

    val ensemb = NodeTemplate(SparkGenericBagging(), Seq(
      LeafTemplate(Bayesian()),
      LeafTemplate(DecisionTree())
    ))

    val hpField = new HyperParametersField( Seq(
      BayesianHPGroup(Seq(Smoothing(Some(6.0)))),
      DecisionTreeHPGroup(Seq(MaxDepth(Some(2.0))))
    ))

    println(TemplateTreeHelper.renderAsString_v2(ensemb))

    val bm = MultiRestartBenchmarker(numberOfRestarts = 10, seed = seed) { (benchmarker: MultiRestartBenchmarker[_], iterationSeed) =>

      val Array(trainingSplit, testSplit) = data.randomSplit(Array(0.67, 0.33), iterationSeed)

      val problemType = ProblemType.MultiClassClassificationProblem

      val baggingF1 = ensemb.evaluateFitness(trainingSplit, testSplit, problemType, hyperParamsField = Some(hpField), seed = iterationSeed).getCorrespondingMetric

      val dtF1 = DecisionTree().fitnessError(trainingSplit, testSplit, problemType, Some(hpField)).getMetricByName("f1") // TODO seed

      println("Bagging's F1:" + baggingF1)
      println("DT's F1:" + dtF1)

      benchmarker.update(baggingF1, dtF1)
    }

    println("Average performance f1: " + bm.avgPerformance)
    println("Average baseline performance f1: " + bm.avgBaselinePerformance)
    bm.avgSuccessRate should be > 0.5
  }
}





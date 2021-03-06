package com.automl.benchmark.glass

import akka.actor.ActorSystem
import com.automl.benchmark.MultiRestartBenchmarker
import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.dataset.Datasets
import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.helper.TemplateTreeHelper
import com.automl.problemtype.ProblemType
import com.automl.spark.SparkSessionProvider
import com.automl.template.{LeafTemplate, NodeTemplate}
import com.automl.template.simple.DecisionTree
import org.apache.spark.sql.DataFrame
import org.scalatest.{BeforeAndAfterAll, FunSuite, Matchers, WordSpec}

import scala.util.Random


// These tests do not use AutoML but directly compute performance on a specific ensembles
class BaggingOnlyDecisionTreesEnsembleSuite extends FunSuite with Matchers with BeforeAndAfterAll with SparkSessionProvider {

  implicit val system = ActorSystem("AutoML-system")

  test("Iris dataset - Spark Bagging should calculate over multiple decision trees( Classification problem ) and berform on average better than single decision tree") {

    val seed = new Random().nextLong()
    println(s"Seed for current test: $seed")
    val data = Datasets.getIrisDataFrame(seed)
    performTestOn(data, seed)

  }

  test("Glass dataset - Spark Bagging should calculate over multiple decision trees( Classification problem ) and perform on average better than single decision tree") {

    val seed = new Random().nextLong()
    println(s"Seed for current test: $seed")
    val data = Datasets.getGlassDataFrame(seed)
    performTestOn(data, seed)

  }

  private def performTestOn(data: DataFrame, seed: Long) = {
    val models = Seq(
      LeafTemplate(DecisionTree()), //TODO We need n-classes +2 base models to be able to find majority
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree())
    )

    val ensemb = NodeTemplate(SparkGenericBagging(), models)

    println(TemplateTreeHelper.renderAsString_v2(ensemb))

    val bm = MultiRestartBenchmarker(numberOfRestarts = 5) { (benchmarker: MultiRestartBenchmarker[_], seed) =>

      val Array(trainingSplit, testSplit) = data.randomSplit(Array(0.67, 0.33), seed)

      val problemType = ProblemType.MultiClassClassificationProblem

      val baggingF1 = ensemb.evaluateFitness(trainingSplit, testSplit, problemType, hyperParamsField = Some(HyperParametersField.default), seed = seed).getCorrespondingMetric

      val dtF1 = DecisionTree().fitnessError(trainingSplit, testSplit, problemType).getMetricByName("f1")

      println("Bagging's F1:" + baggingF1)
      println("DT's F1:" + dtF1)

      benchmarker.update(baggingF1, dtF1)
    }

    bm.avgSuccessRate should be > 0.6
  }
}





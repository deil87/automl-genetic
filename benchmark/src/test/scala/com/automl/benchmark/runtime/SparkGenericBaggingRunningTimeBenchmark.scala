package com.automl.benchmark.runtime

import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.dataset.Datasets
import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.helper.TemplateTreeHelper
import com.automl.problemtype.ProblemType
import com.automl.spark.SparkSessionProvider
import com.automl.template._
import com.automl.template.simple._
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.reflections.scanners.ResourcesScanner
import org.scalatest.{FunSuite, Matchers}
import utils.{BenchmarkHelper, SparkMLUtils}

import scala.util.Random


class SparkGenericBaggingRunningTimeBenchmark extends FunSuite with Matchers with SparkSessionProvider{

  import utils.SparkMLUtils._

  test("Spark Bagging should finish within reasonable time range") {

    val models = Seq(
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree())
    )

    val ensemb = NodeTemplate(SparkGenericBagging(), models)

    println(TemplateTreeHelper.renderAsString_v2(ensemb))

    val shufflingSeed = new Random().nextLong()
    println(s"Shuffling seed for current test: $shufflingSeed")

    val data = Datasets.getIrisDataFrame(shufflingSeed)

    val Array(trainingSplit, testSplit) = data.randomSplit(Array(0.67, 0.33), shufflingSeed)

    val problemType = ProblemType.MultiClassClassificationProblem

    BenchmarkHelper.time("Calculation of f1 by Bagging setimator") {
      val baggingF1 = ensemb.evaluateFitness(trainingSplit, testSplit, problemType, hyperParamsField = Some(HyperParametersField.default), seed = shufflingSeed).getCorrespondingMetric
      println(s"F1 metric: $baggingF1")
    }(0)
  }

}


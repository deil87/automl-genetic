package com.automl.template.simple

import com.automl.ConfigProvider
import com.automl.dataset.Datasets
import com.automl.evolution.dimension.hparameter.{DecisionTreeHPGroup, HyperParametersField, MaxDepth}
import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.spark.SparkSessionProvider
import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.ml.feature.StringIndexer
import org.scalatest.{FunSuite, Matchers}
import utils.SparkMLUtils._
import utils.test.TestBase

import scala.util.Random

class DecisionTreeTest extends FunSuite with SparkSessionProvider with TestBase{

   // unstable as there is no guaranty that it should be true
  ignore("grid search over hyperparameters helps on average") {

    val decisionTreeHPFieldOpt = Some(HyperParametersField(
      Seq(
        DecisionTreeHPGroup(Seq(MaxDepth(Some(3))))
      )
    ))

    def getFitnessForBaselineModel(seed: Long) = {
      ConfigProvider.clearOverride.addOverride(
        """
          |evolution {
          |  hpGridSearch = false
          |  hyperParameterDimension {
          |     enabled = false
          |  }
          |  evaluation {
          |    multiclass.metric = "f1"
          |  }
          |}
        """)

      val dt = DecisionTree(null)

      val preparedGlassDF = Datasets.getGlassDataFrame(seed).sampleRand(50, seed)
//      preparedGlassDF.showAllAndContinue

      val Array(trainingSplit, testSplit) = preparedGlassDF.randomSplit(Array(0.67, 0.33), seed)
      println("Fitness baseline:")
      val f1 = dt.fitnessError(trainingSplit, testSplit, ProblemType.MultiClassClassificationProblem, decisionTreeHPFieldOpt)
      f1
    }


    def getFitnessWithRGS(seed: Long) = {
      ConfigProvider.clearOverride.addOverride(
        """
          |evolution {
          |  hpGridSearch = true
          |  evaluation {
          |    multiclass.metric = "f1"
          |  }
          |}
        """)

      val dt = DecisionTree(null)
      val preparedGlassDF = Datasets.getGlassDataFrame(seed).sampleRand(50, seed)
//      preparedGlassDF.showAllAndContinue

      val Array(trainingSplit, testSplit) = preparedGlassDF.randomSplit(Array(0.67, 0.33), seed)
      println("Fitness with RGS:")
      val f1 = dt.fitnessError(trainingSplit, testSplit, ProblemType.MultiClassClassificationProblem, decisionTreeHPFieldOpt)
      f1
    }

    var avgWithRGS = 0.0
    var avgBaseline = 0.0

    val numberOfRestarts = 3
    for(i <- 0 until numberOfRestarts) {
      val seed = new Random().nextLong()

      val f1: FitnessResult = getFitnessWithRGS(seed)
      avgWithRGS += f1.getCorrespondingMetric

      val f1Baseline: FitnessResult = getFitnessForBaselineModel(seed)

      avgBaseline += f1Baseline.getCorrespondingMetric
    }

    avgWithRGS /= numberOfRestarts
    avgBaseline /= numberOfRestarts
    println(s"avgWithRGS = $avgWithRGS ; avgBaseline = $avgBaseline")
    avgWithRGS should be > avgBaseline

  }



  ignore("testFitnessError") {
    //TODO
  }

  ignore("test for Binary case") {
    //TODO
  }

  test("string indexes generates consistent indexes") {

    import ss.implicits._
    val threeDistinctLevelsDF = ss.sparkContext.parallelize(
      Seq(
        (1, "level1"),
        (2, "level1"),
        (3, "level2"),
        (4, "level3")
      )
    ).toDF("a", "label")

    val threeDistinctLevelsDFInDiffOrder = ss.sparkContext.parallelize(
      Seq(
        (1, "level3"),
        (2, "level2"),
        (3, "level2"),
        (4, "level1")
      )
    ).toDF("a", "label")

    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")

    val labelIndexer1 = indexer
      .fit(threeDistinctLevelsDF.sort("label"))

    val labelIndexer2 = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(threeDistinctLevelsDFInDiffOrder.sort("label"))

    println("Indexes:" + labelIndexer1.labels.mkString(" | "))
    println("Indexes:" + labelIndexer2.labels.mkString(" | "))

    labelIndexer1.labels should not contain theSameElementsInOrderAs(labelIndexer2.labels)
  }

}

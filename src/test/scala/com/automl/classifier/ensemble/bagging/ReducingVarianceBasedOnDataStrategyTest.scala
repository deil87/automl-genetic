package com.automl.classifier.ensemble.bagging

import com.automl.dataset.{Datasets, RandomColumnsSampling, StratifiedRowsSampling}
import com.automl.spark.SparkSessionProvider
import com.automl.template.LeafTemplate
import com.automl.template.simple.{DecisionTree, RandomForest}
import com.test.TestBase
import org.scalatest.FunSuite

class ReducingVarianceBasedOnDataStrategyTest extends FunSuite with SparkSessionProvider with TestBase{


  test("BootstrapingRandomPredictorsRVStrategy should sample columns as well") {

    val rowsSampling = new StratifiedRowsSampling
    val colsSampling = new RandomColumnsSampling

    val irisDF = Datasets.getIrisDataFrame(1234).cache()

    val originalNumberOfColumns = irisDF.columns.size
    val subMembers = Seq(LeafTemplate(DecisionTree()), LeafTemplate(RandomForest()))

    val trainingSamples = new BootstrapingRandomPredictorsRVStrategy(rowsSampling, 0.5, colsSampling, 0.7).generateTrainingSamples(irisDF,  subMembers, None, 2345)

    trainingSamples.forall(sample => sample._2.columns.size < originalNumberOfColumns) shouldBe true
  }
}

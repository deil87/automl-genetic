package com.automl.dataset

import org.scalatest.concurrent.TimeLimitedTests
import org.scalatest.time.{Minutes, Span}
import org.scalatest.{FunSuite, Matchers}
import com.test.TestBase

class DatasetsTest extends FunSuite with TestBase{

  import utils.SparkMLUtils._

  test("Splitting and shuffling works as expected") {
    val seed = 1234
    val seedForShuffling = 3456
    // seed only makes sure the sizes of the splits are constant not the elements inside. seedForShuffling serves for tat purpose
    val Array(trainDF, testDF) = Datasets.getGlassDataFrame(seedForShuffling).randomSplit(Array(0.8, 0.2), seed)
    val Array(trainDF2, testDF2) = Datasets.getGlassDataFrame(seedForShuffling).randomSplit(Array(0.8, 0.2), seed)

    trainDF.count() shouldBe trainDF2.count()

//    trainDF.drop("features").showAll()
//    trainDF2.drop("features").showAll()

    // We can't do comparison as StandardScaler do have rounding errors
    val diff = trainDF.drop("features").except(trainDF2.drop("features"))

//    value.showAll()
    diff.count() shouldBe 0
  }

  test("dataset is shuffled with seed( except works row-wise and order does matter)") {
    val data = Datasets.getIrisDataFrame(1234)
    val data2 = Datasets.getIrisDataFrame(1234)
//    data.showAllAndContinue
//    data2.showAllAndContinue
    data.except(data2).count() shouldBe 0
  }
}

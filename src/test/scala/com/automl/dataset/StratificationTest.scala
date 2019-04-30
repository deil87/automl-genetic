package com.automl.dataset

import org.scalatest.{FunSuite, Matchers}

import scala.util.Random

class StratificationTest extends FunSuite with Matchers{

  test("stratification split is working") {
    val seed = new Random().nextLong()
    val preparedGlassDF = Datasets.getGlassDataFrame(seed)
    preparedGlassDF.cache()

    val Array(training, test) = new Stratification().split(preparedGlassDF, 0.8, seed = seed)
    training.count().toDouble shouldBe preparedGlassDF.count() * 0.8 +- 10.0
  }

}

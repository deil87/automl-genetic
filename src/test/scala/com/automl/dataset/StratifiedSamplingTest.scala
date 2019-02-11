package com.automl.dataset

import org.scalatest.{FunSuite, Matchers}

class StratifiedSamplingTest extends FunSuite with Matchers{

  test("Stratified should return ratio of the data per class") {
    val stratifier = new StratifiedSampling

    val glassDF = Datasets.getGlassDataFrame(1234)
    val sampled = stratifier.sample(glassDF, 0.5)

    val (startValues, counts) = glassDF.select("indexedLabel").rdd.map(value => value.getDouble(0)).histogram(6)
    val (startValuesS, countsS) = sampled.select("indexedLabel").rdd.map(value => value.getDouble(0)).histogram(6)

    counts(0).toDouble / 2 shouldBe countsS(0).toDouble +- 5
  }
}

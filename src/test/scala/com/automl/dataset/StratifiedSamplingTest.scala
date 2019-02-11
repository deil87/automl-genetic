package com.automl.dataset

import org.scalatest.FunSuite

class StratifiedSamplingTest extends FunSuite {

  test("Stratified should return ratio of the data per class") {
    val stratifier = new StratifiedSampling

    val glassDF = Datasets.getGlassDataFrame(1234)
    stratifier.sample(glassDF, (glassDF.count() * 0.8).toLong)
  }
}

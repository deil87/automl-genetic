package com.automl.evolution.dimension

import org.scalatest.{FunSuite, Matchers}

class TemplateHyperParametersEvolutionDimensionTest extends FunSuite with Matchers{

  test("mutation of the group should work") {
    val hpGroupOfParameters =  BayesianHPGroup()
    val smoothing = hpGroupOfParameters.hpParameters
    val hpGroupOfParametersOrig = hpGroupOfParameters.mutate()

    val isAllParametersChanged = hpGroupOfParametersOrig.hpParameters.forall{ case param@Smoothing() =>
      param.currentValue != smoothing
    }
    isAllParametersChanged should be(true)
  }
}

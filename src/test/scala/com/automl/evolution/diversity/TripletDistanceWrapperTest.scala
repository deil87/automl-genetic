package com.automl.evolution.diversity

import org.scalatest.{FunSuite, Matchers}
import utils.test.TestBase

class TripletDistanceWrapperTest extends FunSuite with TestBase{

  test("testComputeDistance") {

    new TripletDistanceWrapper().computeDistance should be(0)
  }

}

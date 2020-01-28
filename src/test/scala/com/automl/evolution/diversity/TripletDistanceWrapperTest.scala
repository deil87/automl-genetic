package com.automl.evolution.diversity

import org.scalatest.{FunSuite, Matchers}
import com.test.TestBase

class TripletDistanceWrapperTest extends FunSuite with TestBase{

  test("testComputeDistance") {

    new TripletDistanceWrapper().computeDistance should be(0)
  }

}

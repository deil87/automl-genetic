package com.automl.evolution.diversity

import org.scalatest.{FunSuite, Matchers}

class TripletDistanceWrapperTest extends FunSuite with Matchers{

  test("testComputeDistance") {

    new TripletDistanceWrapper().computeDistance should be(0)
  }

}

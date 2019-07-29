package com.automl.evolution.dimension.hparameter

import org.scalatest.{FunSuite, Matchers}

class DoubleHPRangeTest extends FunSuite with Matchers{

  test("testGetNextWithinTheRange method works as expected") {
    val maxDepth = MaxDepth()
    1 to 1000 forall (idx => {
      println("Iteratino: " + idx)
      val next = maxDepth.getNextWithinTheRange
      next <= maxDepth.max && next >= maxDepth.min
    }) shouldBe true
  }

}

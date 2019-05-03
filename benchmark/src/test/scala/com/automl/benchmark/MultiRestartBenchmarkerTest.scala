package com.automl.benchmark

import org.scalatest.{FunSuite, Matchers}

import scala.util.Random

class MultiRestartBenchmarkerTest extends FunSuite with Matchers{

  test("testApply") {

    val res = MultiRestartBenchmarker(numberOfRestarts = 10000) { benchmarker: MultiRestartBenchmarker[_] =>

      val specific = new Random().nextDouble() + 0.2
      val baseline = new Random().nextDouble()
      benchmarker.update(specific, baseline)

    }

    res.avgSuccessRate shouldBe(0.7 +- 0.05)
  }

}

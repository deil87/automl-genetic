package com.automl.template.simple

import org.scalatest.concurrent.DoNotInterrupt
import org.scalatest.{FunSpec, FunSuite}
import org.scalatest.concurrent.TimeLimitedTests
import org.scalatest.time.Span
import org.scalatest.time.SpanSugar._

class TimeLimitedTest extends FunSuite with TimeLimitedTests {

  val timeLimit: Span = 200 millis

//  override val defaultTestInterruptor = DoNotInterrupt

//  describe("A time-limited test") {
//    it("should succeed if it completes within the time limit") {
//      Thread.sleep(100)
//    }
//    it("should fail if it is taking too darn long") {
//      Thread.sleep(300)
//    }
//  }

  test("should check that size of input layer correlates with number of features") {
      Thread.sleep(300)
  }
}

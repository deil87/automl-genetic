package com.test

import com.typesafe.scalalogging.LazyLogging
import org.scalatest.concurrent.TimeLimitedTests
import org.scalatest.time.{Minutes, Seconds, Span}
import org.scalatest.{Matchers, TestSuite}

trait TestBase extends Matchers with LazyLogging with TimeLimitedTests{  this: TestSuite =>

  val timeLimit: Span = Span(9, Minutes)

}

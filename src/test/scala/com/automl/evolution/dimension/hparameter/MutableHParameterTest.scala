package com.automl.evolution.dimension.hparameter

import com.typesafe.scalalogging.LazyLogging
import org.scalatest.{FunSuite, Matchers}
import com.test.TestBase

import scala.collection.mutable

class MutableHParameterTest extends FunSuite with TestBase {

  test("Mutation should return only new values") {
    var smoothing = Smoothing()
    val firstValue = smoothing.currentValue
    val newValue = smoothing.mutate()
    firstValue == newValue.currentValue shouldBe false

    smoothing.numberOfEntries shouldEqual 10

    1 to 50 foreach { _ =>
      ignoreControllableException {
        smoothing = smoothing.mutate().asInstanceOf[Smoothing]
      }
      logger.debug("Smoothing:" + smoothing.currentValue)
    }
  }

  test ("Eventually we explore all the values from the range. Also checking that values are within range.") {
    val explored = mutable.Map.empty[Double, Boolean]

    var elasticNet = ElasticNet()
    elasticNet.numberOfEntries shouldEqual 11

    1 to 1000 foreach { _ =>
      ignoreControllableException {
        elasticNet = elasticNet.mutate().asInstanceOf[ElasticNet]
      }
      explored(elasticNet.currentValue) = true
      logger.debug("ElasticNet:" + elasticNet.currentValue)

      elasticNet.currentValue <= elasticNet.max && elasticNet.currentValue >= elasticNet.min shouldBe true
    }

    explored.size == elasticNet.numberOfEntries shouldBe true
  }

  test ("We don't have repetitions until we explored everything") {
   // TODO we need a `explored` cache to remember what was already exlored
  }


  // This type of tests are flaky as there is a chance for them not to explore all range due to randoness. We can just choose one direction for mutation.
  test("Mutation of Smoothing should return only new values") {
    var smoothing = Smoothing()
    val firstValue = smoothing.currentValue
    val newValue = smoothing.mutate()
    firstValue == newValue.currentValue shouldBe false

    smoothing.numberOfEntries shouldEqual 10

    1 to 50 foreach { _ =>
      ignoreControllableException {
        smoothing = smoothing.mutate().asInstanceOf[Smoothing]
      }
      logger.debug("Smoothing:" + smoothing.currentValue)
    }
  }

  test("Mutation of ElasticNet should return only new values") {
    var elasticNet = ElasticNet()
    val firstValue = elasticNet.currentValue
    val newValue = elasticNet.mutate()
    firstValue == newValue.currentValue shouldBe false

    elasticNet.numberOfEntries shouldEqual 11

    1 to 50 foreach { _ =>
      ignoreControllableException {
        elasticNet = elasticNet.mutate().asInstanceOf[ElasticNet]
      }
      logger.debug("ElasticNet:" + elasticNet.currentValue)
    }
  }

  //TODO AG-133
  test("Mutation of LRRegParam should return only new values") {
    var regParam = RegParamLR()
    val firstValue = regParam.currentValue
    val newValue = regParam.mutate()
    firstValue == newValue.currentValue shouldBe false

    regParam.numberOfEntries shouldEqual 6

    1 to 50 foreach { _ =>
      ignoreControllableException {
        regParam = regParam.mutate().asInstanceOf[RegParamLR]
      }
      logger.debug("LRRegParam:" + regParam.currentValue)
    }
  }

  test("Mutation of MaxDepth should return only new values") {
    var maxDepth = MaxDepth()
    val firstValue = maxDepth.currentValue
    val newValue = maxDepth.mutate()
    firstValue == newValue.currentValue shouldBe false

    maxDepth.numberOfEntries shouldEqual 6

    1 to 50 foreach { _ =>
      ignoreControllableException {
        maxDepth = maxDepth.mutate().asInstanceOf[MaxDepth]
      }
      logger.debug("MaxDepth:" + maxDepth.currentValue)
    }
  }

  test ("Eventually we will explore all the values from the range of MaxDepth. Also checking that values are within range.") {
    val explored = mutable.Map.empty[Double, Boolean]

    var maxDepth = MaxDepth()
    maxDepth.numberOfEntries shouldEqual 6

    1 to 10000 foreach { _ =>
      ignoreControllableException {
        maxDepth = maxDepth.mutate().asInstanceOf[MaxDepth]
      }
      explored(maxDepth.currentValue) = true
      logger.debug("MaxDepth:" + maxDepth.currentValue)

      maxDepth.currentValue <= maxDepth.max && maxDepth.currentValue >= maxDepth.min shouldBe true
    }

    explored.size == maxDepth.numberOfEntries shouldBe true
  }

  def ignoreControllableException(fun: => Unit): Unit = {
    try {
      fun
    } catch {
      case ex: HPRangeWasExploredException =>
    }
  }

}

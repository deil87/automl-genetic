package com.automl.evolution.dimension.hparameter

import com.typesafe.scalalogging.LazyLogging
import org.scalatest.{FunSuite, Matchers}

import scala.collection.mutable

class DoubleHPRangeTest extends FunSuite with Matchers with LazyLogging{

  test("testGetNextWithinTheRange method works as expected") {
    val maxDepth = MaxDepth()
    logger.debug("Max boundary:" + maxDepth.max)
    logger.debug("Min boundary:" + maxDepth.min)

    val occurancesMap = mutable.Map.empty[Double, Int]

    1 to 1000 forall (idx => {
      val next = maxDepth.getNextWithinTheRange
      logger.debug("Iteration: " + idx + " -> " + next)

      val currentCounterValue = occurancesMap.getOrElseUpdate(next, 0)
      occurancesMap.put(next, currentCounterValue + 1)
      next <= maxDepth.max && next >= maxDepth.min
    }) shouldBe true

    logger.debug("Histogram:" + occurancesMap.toString())
  }

  test("testGetNextWithinTheRange method works as expected when step is less then 1") {
    val regParam = LRRegParam()
    logger.debug("Max boundary:" + regParam.max)
    logger.debug("Min boundary:" + regParam.min)
    val occurancesMap = mutable.Map.empty[Double, Int]
    1 to 10000 forall (idx => {
      val next = regParam.getNextWithinTheRange
      logger.debug("Iteration: " + idx + " -> " + next)

      val currentCounterValue = occurancesMap.getOrElseUpdate(next, 0)
      occurancesMap.put(next, currentCounterValue + 1)

      next <= regParam.max && next >= regParam.min
    }) shouldBe true

    logger.debug("Histogram:" + occurancesMap.toString())
  }

  // But with random jump it always actually can. Need to relax requirement
  test("test that mutate() will not return the same value after mutation") {
    val regParam = LRRegParam()
    logger.debug("Max boundary:" + regParam.max)
    logger.debug("Min boundary:" + regParam.min)
    1 to 10000 foreach (idx => {
      val before = regParam.currentValue
      val next = regParam.mutate()
      logger.debug("Iteration: " + idx + " -> " + next)

      before != next.currentValue shouldBe true
    })

  }

}

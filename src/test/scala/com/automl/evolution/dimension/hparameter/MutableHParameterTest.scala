package com.automl.evolution.dimension.hparameter

import org.scalatest.{FunSuite, Matchers}

import scala.collection.mutable

class MutableHParameterTest extends FunSuite with Matchers{

  test("Mutation should return only new values") {
    var smoothing = Smoothing()
    val firstValue = smoothing.currentValue
    val newValue = smoothing.mutate()
    firstValue == newValue.currentValue shouldBe false

    smoothing.numberOfEntries shouldEqual 10

    1 to 50 foreach { _ =>
      smoothing = smoothing.mutate()
      println(smoothing.currentValue)
    }
  }

  test ("Eventually we explore all the values from the range. Also checking that values are within range.") {
    val explored = mutable.Map.empty[Double, Boolean]

    var elasticNet = ElasticNet()
    elasticNet.numberOfEntries shouldEqual 11

    1 to 1000 foreach { _ =>
      elasticNet = elasticNet.mutate()
      explored(elasticNet.currentValue) = true
      println(elasticNet.currentValue)

      elasticNet.currentValue <= elasticNet.max && elasticNet.currentValue >= elasticNet.min shouldBe true
    }

    explored.size == elasticNet.numberOfEntries shouldBe true
  }

  test ("We don't have repetitions until we explored everything") {
   // TODO we need a `explored` cache to remember what was already exlored
  }


  test("Mutation of ElasticNet should return only new values") {
    var elasticNet = ElasticNet()
    val firstValue = elasticNet.currentValue
    val newValue = elasticNet.mutate()
    firstValue == newValue.currentValue shouldBe false

    elasticNet.numberOfEntries shouldEqual 11

    1 to 50 foreach { _ =>
      elasticNet = elasticNet.mutate()
      println(elasticNet.currentValue)
    }
  }

  //TODO AG-133
  test("Mutation of LRRegParam should return only new values") {
    var regParam = LRRegParam()
    val firstValue = regParam.currentValue
    val newValue = regParam.mutate()
    firstValue == newValue.currentValue shouldBe false

    regParam.numberOfEntries shouldEqual 11

    1 to 50 foreach { _ =>
      regParam = regParam.mutate()
      println(regParam.currentValue)
    }
  }

  test("Mutation of MaxDepth should return only new values") {
    var maxDepth = MaxDepth()
    val firstValue = maxDepth.currentValue
    val newValue = maxDepth.mutate()
    firstValue == newValue.currentValue shouldBe false

    maxDepth.numberOfEntries shouldEqual 10

    1 to 50 foreach { _ =>
      maxDepth = maxDepth.mutate()
      println(maxDepth.currentValue)
    }
  }
}
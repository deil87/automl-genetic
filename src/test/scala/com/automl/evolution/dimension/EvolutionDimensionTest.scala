package com.automl.evolution.dimension

import org.scalatest.{FunSuite, Matchers}

import scala.collection.mutable

class EvolutionDimensionTest extends FunSuite with Matchers {

  test("priority queue is being appended") {
    val hallOfFame = new mutable.PriorityQueue[Double]()
    hallOfFame ++= Seq(42.0)
    hallOfFame.head should be(42.0)
  }
}

package com

import org.scalatest.{FunSuite, Matchers}

import scala.util.control.NonFatal


class ShapelessSuite extends FunSuite with Matchers{

  import shapeless._
  import HList._
  import syntax.std.traversable._

  test("Combine tuples with shapeless") {
    val x = List(1, 2, 3)
    val y = x.toHList[Int::Int::Int::HNil]
    val z = y.get.tupled

    val tup = (42, 43)
    import syntax.std.tuple._
    val res = tup ++ z

    res should be ((42,43,1,2,3))

  }

}



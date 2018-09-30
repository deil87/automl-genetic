package com.automl

import org.scalatest.{Matchers, WordSpec}

class AutoMLPackageObjectTest extends WordSpec with Matchers{

  case class Item(name: String)
  "automl package object's methods" should {

    "extract random elements from the list and return tuple of (randelem, rest)" in {

      val items = List((Item("0.2"), 0.2), (Item("0.8"), 0.8))
      val (extractedRand, rest ) = items.extractElem

      rest :+ extractedRand should contain theSameElementsAs(items)
    }

  }
}

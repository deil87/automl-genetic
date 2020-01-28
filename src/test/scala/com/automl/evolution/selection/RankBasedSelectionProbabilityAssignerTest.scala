package com.automl.evolution.selection

import org.scalatest.{Matchers, WordSpec}
import com.test.TestBase

class RankBasedSelectionProbabilityAssignerTest extends WordSpec with TestBase{

  case class Item(name: String)
  "RankBasedSelectionProbabilityAssignerTest" should {

    "get assigned probabilities for two items" in {

      val items = List(0.2,  0.8)

      val assigned =  new RankBasedSelectionProbabilityAssigner[Double].assign(items)


      assigned(0)._2 shouldBe 0.0 +- 0.01
      assigned(1)._2 shouldBe 1.0 +- 0.01
    }

    "get assigned probabilities for tree items" in {

      val items = List(0.2, 0.3, 0.5)

      val assigned =  new RankBasedSelectionProbabilityAssigner[Double].assign(items)


      assigned(0)._2 shouldBe 0.0 +- 0.01
      assigned(1)._2 shouldBe 0.33 +- 0.01
      assigned(2)._2 shouldBe 0.67 +- 0.01
    }

    "get assigned probabilities for five items" in {

      val items = List(0.2, 0.3, 0.4, 0.5, 0.9)

      val assigned =  new RankBasedSelectionProbabilityAssigner[Double].assign(items)


      assigned(0)._2 shouldBe 0.0 +- 0.01
      assigned(1)._2 shouldBe 0.1 +- 0.01
      assigned(2)._2 shouldBe 0.2 +- 0.01
      assigned(3)._2 shouldBe 0.3 +- 0.01
      assigned(4)._2 shouldBe 0.4 +- 0.01
    }

    "get assigned probabilities for list of one element" in {

      val items = List(0.2)

      val assigned =  new RankBasedSelectionProbabilityAssigner[Double].assign(items)

      assigned(0)._2 shouldBe 1.0 +- 0.01
    }

  }
}

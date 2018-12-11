package com.automl.evolution.selection

import com.automl.helper.{FitnessResult, PopulationHelper}
import com.automl.template.LeafTemplate
import com.automl.template.simple.{DecisionTree, LinearRegressionModel, SimpleModelMember}
import com.automl.{EvaluatedTemplateData, Population}
import org.scalatest.{Matchers, WordSpec}

import scala.util.Random

class RouletteWheelSelectorTest extends WordSpec with Matchers{

  case class Item(name: String)
  "CumulativeProbabilitySelectorTest" should {

    "get next element from the list based on the assigned probabilities" in {

      val items = List((Item("0.2"), 0.2), (Item("0.8"), 0.8))
      val selector = new RouletteWheel(items)

      var counter2 = 0.0
      var counter8 = 0.0
      val totalNumberOfDraws = 10000

      (1 to totalNumberOfDraws).foreach{ _ =>
        selector.getNext match {
          case (Item("0.2"), _) => counter2+=1
          case (Item("0.8"), _) => counter8+=1
        }
      }

      val actualProbabilityFor2 = counter2 / totalNumberOfDraws
      val actualProbabilityFor8 = counter8 / totalNumberOfDraws
      println("Probability for Item(0.2) is " + actualProbabilityFor2)
      println("Probability for Item(0.8) is " + actualProbabilityFor8)

      actualProbabilityFor2 shouldBe 0.2 +- 0.01
      actualProbabilityFor8 shouldBe 0.8 +- 0.01
    }

    "check validation that requires probabilities to add up to unity (one)" in {

      val items = List((Item("0.3"), 0.3), (Item("0.8"), 0.8))

      assertThrows[IllegalArgumentException] {
        val selector = new RouletteWheel(items)
      }
    }
  }
}

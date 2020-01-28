package com.automl.evolution.selection

import com.automl.helper.{FitnessResult, PopulationHelper}
import com.automl.template.LeafTemplate
import com.automl.template.simple.{DecisionTree, LinearRegressionModel, SimpleModelMember}
import com.automl.EvaluatedTemplateData
import com.automl.problemtype.ProblemType.MultiClassClassificationProblem
import org.scalatest.{Matchers, WordSpec}
import com.test.TestBase

import scala.util.Random

class RouletteWheelSelectorTest extends WordSpec with TestBase{

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

    "get sample element from the list based on the assigned probabilities" in {

      val items = List((Item("0.2"), 0.2), (Item("0.8"), 0.8))
      val selector = new RouletteWheel(items)

      var counter2 = 0.0
      var counter8 = 0.0
      val totalNumberOfDraws = 10000

      selector.sample(totalNumberOfDraws).foreach {
        case (Item("0.2"), _) => counter2 += 1
        case (Item("0.8"), _) => counter8 += 1
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


    "sorting to get rank of the items is done based on the input probabilities" in {

      val evaluatedA: (EvaluatedTemplateData, Double) = (EvaluatedTemplateData("id1", null, null, FitnessResult(Map("f1" -> 0.2, "logloss" -> 0.3), MultiClassClassificationProblem, null)), 0.2)
      val evaluatedB = (EvaluatedTemplateData("id2", null, null, FitnessResult(Map("f1" -> 0.3, "logloss" -> 0.2), MultiClassClassificationProblem, null)), 0.3)
      val evaluatedC = (EvaluatedTemplateData("id3", null, null, FitnessResult(Map("f1" -> 0.1, "logloss" -> 0.1), MultiClassClassificationProblem, null)), 0.5)

      val items = List(evaluatedA, evaluatedB,evaluatedC)

      val selector = new RouletteWheel[EvaluatedTemplateData](items)

      selector.getSortedItemsASC.head shouldBe evaluatedA
    }
  }
}

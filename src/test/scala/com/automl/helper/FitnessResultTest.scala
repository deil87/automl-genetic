package com.automl.helper

import com.automl.ConfigProvider
import com.automl.problemtype.ProblemType.MultiClassClassificationProblem
import org.scalatest.{FunSuite, Matchers}
import utils.TestBase

class FitnessResultTest extends FunSuite with TestBase {

  test("betterThan for FitnessResult takes into account metric") {

    ConfigProvider.clearOverride.addOverride(
      """
        |evolution {
        |  evaluation {
        |    multiclass.metric = "f1"
        |  }
        |}
      """)

    val fr1 = FitnessResult(Map("f1" -> 0.9, "logloss" -> 0.32), MultiClassClassificationProblem, null)
    val fr2 = FitnessResult(Map("f1" -> 0.8, "logloss" -> 0.11), MultiClassClassificationProblem, null)
    fr1.betterThan(fr2) shouldBe true

    ConfigProvider.clearOverride.addOverride(
      """
        |evolution {
        |  evaluation {
        |    multiclass.metric = "logloss"
        |  }
        |}
      """)

    fr1.betterThan(fr2) shouldBe false
  }

  test("betterThanOrEqual for FitnessResult takes into account metric") {

    ConfigProvider.clearOverride.addOverride(
      """
        |evolution {
        |  evaluation {
        |    multiclass.metric = "f1"
        |  }
        |}
      """)

    val fr1 = FitnessResult(Map("f1" -> 0.9, "logloss" -> 0.32), MultiClassClassificationProblem, null)
    val fr2 = FitnessResult(Map("f1" -> 0.8, "logloss" -> 0.11), MultiClassClassificationProblem, null)
    val fr3 = FitnessResult(Map("f1" -> 0.8, "logloss" -> 0.11), MultiClassClassificationProblem, null)
    fr1.betterThanOrEqual(fr2) > 0 shouldBe true
    fr2.betterThanOrEqual(fr3) == 0 shouldBe true

    ConfigProvider.clearOverride.addOverride(
      """
        |evolution {
        |  evaluation {
        |    multiclass.metric = "logloss"
        |  }
        |}
      """)

    fr2.betterThanOrEqual(fr1) > 0 shouldBe true
    fr3.betterThanOrEqual(fr2) == 0 shouldBe true
    fr1.betterThanOrEqual(fr2) < 0 shouldBe true
  }

  test("that Ordering is matching conventions") {
    val listOfNumbers = collection.mutable.PriorityQueue[Double]()
    val list = Seq(42.0, 36.0, 75.0)
    listOfNumbers.enqueue(42.0, 36.0, 75.0, 22.0)
    listOfNumbers.head should be (75.0)

    ConfigProvider.clearOverride.addOverride(
      """
        |evolution {
        |  evaluation {
        |    multiclass.metric = "f1"
        |  }
        |}
      """)

    val fr1 = FitnessResult(Map("f1" -> 0.9, "logloss" -> 0.32), MultiClassClassificationProblem, null)
    val fr2 = FitnessResult(Map("f1" -> 0.7, "logloss" -> 0.11), MultiClassClassificationProblem, null)
    val fr3 = FitnessResult(Map("f1" -> 0.7, "logloss" -> 0.11), MultiClassClassificationProblem, null)

    val hallOfFameBasedOnf1 = collection.mutable.PriorityQueue[FitnessResult]()

    hallOfFameBasedOnf1.enqueue(fr2, fr1, fr3)
    hallOfFameBasedOnf1.head shouldBe fr1

    ConfigProvider.clearOverride.addOverride(
      """
        |evolution {
        |  evaluation {
        |    multiclass.metric = "logloss"
        |  }
        |}
      """)

    val hallOfFameBasedOnLogloss = collection.mutable.PriorityQueue[FitnessResult]()

    hallOfFameBasedOnLogloss.enqueue(fr2, fr1, fr3)
    hallOfFameBasedOnLogloss.head shouldBe fr2
  }

  // By default sorted use ASCENDING. But for logloss smaller the better and Ascending would mean from worse(big logloss) towards best(smallest logloss)
  test("prove that list and priority queue have different default orders") {
    val list = Seq(42.0, 36.0, 75.0)
    list.sorted.head should be (36.0)

    // By default head() returns best element straight away
    val listOfNumbers = collection.mutable.PriorityQueue[Double]()
    listOfNumbers.enqueue(list:_*)
    listOfNumbers.head shouldBe 75.0
  }

  test("order for FitnessResult with `logloss` metric is ascending ( the less the better case)") {
    ConfigProvider.clearOverride.addOverride(
      """
        |evolution {
        |  evaluation {
        |    multiclass.metric = "logloss"
        |  }
        |}
      """)

    val bestFR = FitnessResult(Map("logloss" -> 36.0), MultiClassClassificationProblem, null)
    val fitnessResults = Seq(FitnessResult(Map("logloss" -> 42.0), MultiClassClassificationProblem, null),
      bestFR,
      FitnessResult(Map("logloss" -> 75.0), MultiClassClassificationProblem, null)
    )
    // By default sorted use ASCENDING. But for logloss smaller the better and Ascending would mean from worse(big logloss) towards best(smallest logloss)
    fitnessResults.sorted.reverse.head shouldBe bestFR

    val hallOfFameBasedOnLogloss = collection.mutable.PriorityQueue[FitnessResult]()

    hallOfFameBasedOnLogloss.enqueue(fitnessResults:_*)
    hallOfFameBasedOnLogloss.head shouldBe bestFR
  }

  test("order for FitnessResult with `f1` metric is descending for sorted list ( the greater the better case)") {

    ConfigProvider.clearOverride.addOverride(
      """
        |evolution {
        |  evaluation {
        |    multiclass.metric = "f1"
        |  }
        |}
      """)

    val bestFR = FitnessResult(Map("f1" -> 75.0), MultiClassClassificationProblem, null)
    val fitnessResults = Seq(FitnessResult(Map("f1" -> 42.0), MultiClassClassificationProblem, null),
      FitnessResult(Map("f1" -> 36.0), MultiClassClassificationProblem, null),
      bestFR
    )

    fitnessResults.sorted.reverse.head shouldBe bestFR

    val hallOfFameBasedOnLogloss = collection.mutable.PriorityQueue[FitnessResult]()

    hallOfFameBasedOnLogloss.enqueue(fitnessResults:_*)
    hallOfFameBasedOnLogloss.head shouldBe bestFR
  }
}

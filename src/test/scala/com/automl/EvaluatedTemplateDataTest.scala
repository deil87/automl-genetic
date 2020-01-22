package com.automl

import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType.MultiClassClassificationProblem
import com.typesafe.config.{Config, ConfigFactory, ConfigRenderOptions}
import org.scalatest.{FunSuite, Matchers}

class EvaluatedTemplateDataTest extends FunSuite with Matchers {

  test("conventional ordering") {
    val list = Seq(42.0, 36.0, 75.0)
    list.sorted.head should be (36.0)
  }

  // But the issue is with `implicit val impl = implicitly[Ordering[EvaluatedTemplateData]]`. It stays in scope and seems like is not being overriden after first setting
  test("config for EvaluationRules is being changed correctly") {
    ConfigProvider.clearOverride.addOverride(
      """
        |evolution {
        |  evaluation {
        |    multiclass.metric = "f1"
        |  }
        |}
      """)

    val evaluatedA = EvaluatedTemplateData("id1", null, null, FitnessResult(Map("f1" -> 0.2, "logloss" -> 0.3), MultiClassClassificationProblem, null))
    evaluatedA.theBiggerTheBetter(MultiClassClassificationProblem) shouldBe true

    ConfigProvider.clearOverride.addOverride(
      """
        |evolution {
        |  evaluation {
        |    multiclass.metric = "logloss"
        |  }
        |}
      """)

    evaluatedA.theBiggerTheBetter(MultiClassClassificationProblem) shouldBe false

  }

  test("ordering for EvaluatedTemplateData respects configuration parameters as well") {
    ConfigProvider.clearOverride.addOverride(
      """
        |evolution {
        |  evaluation {
        |    multiclass.metric = "f1"
        |  }
        |}
      """)

    val evaluatedA = EvaluatedTemplateData("id1", null, null, FitnessResult(Map("f1" -> 0.2, "logloss" -> 0.3), MultiClassClassificationProblem, null))
    val evaluatedB = EvaluatedTemplateData("id2", null, null, FitnessResult(Map("f1" -> 0.3, "logloss" -> 0.2), MultiClassClassificationProblem, null))
    val evaluatedC = EvaluatedTemplateData("id3", null, null, FitnessResult(Map("f1" -> 0.1, "logloss" -> 0.1), MultiClassClassificationProblem, null))

    implicit val impl = implicitly[Ordering[EvaluatedTemplateData]]

    val sortedItems = Seq(evaluatedA, evaluatedB,evaluatedC).sorted

    evaluatedB.betterThan(evaluatedA) shouldBe true

    evaluatedB.compare(evaluatedA) > 0 shouldBe true

    sortedItems.head should be (evaluatedC)
    sortedItems.reverse.head should be (evaluatedB)
  }

  test("ordering for EvaluatedTemplateData respects configuration parameters logloss") {
    ConfigProvider.clearOverride.addOverride(
      """
        |evolution {
        |  evaluation {
        |    multiclass.metric = "logloss"
        |  }
        |}
      """)

    val evaluatedA = EvaluatedTemplateData("id1", null, null, FitnessResult(Map("logloss" -> 0.3), MultiClassClassificationProblem, null))
    val evaluatedB = EvaluatedTemplateData("id2", null, null, FitnessResult(Map("logloss" -> 0.2), MultiClassClassificationProblem, null))
    val evaluatedC = EvaluatedTemplateData("id3", null, null, FitnessResult(Map("logloss" -> 0.1), MultiClassClassificationProblem, null))

    implicit val impl = implicitly[Ordering[EvaluatedTemplateData]]

    val sortedBasedOnLoglossItems = Seq(evaluatedA, evaluatedB,evaluatedC).sorted(impl)
    sortedBasedOnLoglossItems.head shouldBe evaluatedA
    sortedBasedOnLoglossItems.reverse.head should be (evaluatedC)
  }
}

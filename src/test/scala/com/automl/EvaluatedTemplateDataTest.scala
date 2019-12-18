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

  test("order for FitnessResult with `logloss` metric is ascending ( the less the better case)") {
    val testOverride: Config = ConfigFactory.parseString(
      """
        |evolution {
        |  evaluation {
        |    multiclass.metric = "logloss"
        |  }
        |}
      """.stripMargin)
    ConfigProvider.addOverride(testOverride)

    val list = Seq(FitnessResult(Map("logloss" -> 42.0), MultiClassClassificationProblem, null),
      FitnessResult(Map("logloss" -> 36.0), MultiClassClassificationProblem, null),
      FitnessResult(Map("logloss" -> 75.0), MultiClassClassificationProblem, null)
    )
    list.sorted.head.getMetricByName("logloss") should be (36.0)
  }

  test("order for FitnessResult with `f1` metric is descending ( the greater the better case)") {

    ConfigProvider.addOverride(
      """
        |evolution {
        |  evaluation {
        |    multiclass.metric = "f1"
        |  }
        |}
      """)

    val list = Seq(FitnessResult(Map("f1" -> 42.0), MultiClassClassificationProblem, null),
      FitnessResult(Map("f1" -> 36.0), MultiClassClassificationProblem, null),
      FitnessResult(Map("f1" -> 75.0), MultiClassClassificationProblem, null)
    )
    list.sorted.head.getMetricByName("f1") should be (75.0)
  }

  test("ordering for EvaluatedTemplateData respects configuration parameters as well") {
    val testOverride: Config = ConfigFactory.parseString(
      """
        |evolution {
        |  evaluation {
        |    multiclass.metric = "f1"
        |  }
        |}
      """.stripMargin)
    ConfigProvider.addOverride(testOverride)

    val evaluatedA = new EvaluatedTemplateData("id1", null, null, FitnessResult(Map("f1" -> 5.2), MultiClassClassificationProblem, null))
    val evaluatedB = new EvaluatedTemplateData("id2", null, null, FitnessResult(Map("f1" -> 5.3), MultiClassClassificationProblem, null))
    val evaluatedC = new EvaluatedTemplateData("id3", null, null, FitnessResult(Map("f1" -> 5.1), MultiClassClassificationProblem, null))

    val sortedItems = Seq(evaluatedA, evaluatedB,evaluatedC).sorted

    sortedItems.head.result.getMetricByName("f1") should be (5.3)
  }
}

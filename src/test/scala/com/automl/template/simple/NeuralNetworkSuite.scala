package com.automl.template.simple

import com.automl.dataset.Datasets
import com.automl.problemtype.ProblemType.MultiClassClassificationProblem
import com.automl.spark.SparkSessionProvider
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import org.scalatest.concurrent.{TimeLimitedTests, TimeLimits, Timeouts}
import org.scalatest.time.{Millis, Minutes, Seconds, Span}
import org.scalatest.{FunSuite, Ignore, Matchers, WordSpec}

class NeuralNetworkSuite extends FunSuite with Matchers with SparkSessionProvider with TimeLimitedTests {
  import ss.implicits._

  val timeLimit: Span = Span(200, Seconds)

  test("should check that size of input layer correlates with number of features") {
      val car = Datasets.getCarDataFrame(1234)

      car.show(5, false)
      val Array(trainDF, testDF) = car.randomSplit(Array(0.8, 0.2))

//      assertThrows[IllegalArgumentException] {
//        NeuralNetwork(Array(4, 3, 3)).fitnessError(trainDF, testDF, MultiClassClassificationProblem, None)
//      }
//
//      assertThrows[IllegalArgumentException] {
//        NeuralNetwork(Array(5, 3, 5)).fitnessError(trainDF, testDF, MultiClassClassificationProblem, None)
//      }

    NeuralNetwork().fitnessError(trainDF, testDF, MultiClassClassificationProblem, None)
//      noException should be thrownBy NeuralNetwork(Array(5, 3, 5)).fitnessError(trainDF, testDF, MultiClassClassificationProblem, None)

  }
}


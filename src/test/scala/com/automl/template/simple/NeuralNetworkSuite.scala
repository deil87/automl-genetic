package com.automl.template.simple

import com.automl.dataset.Datasets
import com.automl.problemtype.ProblemType.MultiClassClassificationProblem
import com.automl.spark.SparkSessionProvider
import org.scalatest.{FunSuite, Ignore, Matchers, WordSpec}
import utils.TestBase

class NeuralNetworkSuite extends FunSuite with SparkSessionProvider with TestBase {
  import ss.implicits._

  test("should be able to train NeuralNetwork basic scenario") {
      val car = Datasets.getCarDataFrame(1234)

      car.show(5, false)
      val Array(trainDF, testDF) = car.randomSplit(Array(0.8, 0.2))

      NeuralNetwork().fitnessError(trainDF, testDF, MultiClassClassificationProblem, None)

  }
}


package com.automl.template.simple

import com.automl.dataset.Datasets
import com.automl.problemtype.ProblemType
import com.automl.spark.SparkSessionProvider
import org.scalatest.{FunSuite, Matchers}

class BayesianTest extends FunSuite with SparkSessionProvider with Matchers{



  val problemType = ProblemType.MultiClassClassificationProblem

  test("testFitnessError") {
    val Array(trainDF, testDF) = Datasets.getWineDataframe.randomSplit(Array(0.8, 0.2))
    trainDF.cache()
    testDF.cache()

    val bayesianF1 = Bayesian().fitnessError(trainDF, testDF, problemType).getCorrespondingMetric
    println(s"F1 computed for Bayesian model $bayesianF1")

    bayesianF1 shouldBe 0.9 +- 0.1
  }

  test("testFitnessError with Glass dataset") {
    val Array(trainDF, testDF) = Datasets.getGlassDataFrame.randomSplit(Array(0.8, 0.2))
    trainDF.cache()
    testDF.cache()

    val bayesianF1 = Bayesian().fitnessError(trainDF, testDF, problemType).getCorrespondingMetric
    println(s"F1 computed for Bayesian model $bayesianF1")

    bayesianF1 shouldBe 0.9 +- 0.1
  }

}

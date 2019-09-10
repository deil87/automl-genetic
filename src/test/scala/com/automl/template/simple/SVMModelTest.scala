package com.automl.template.simple

import com.automl.dataset.Datasets
import com.automl.problemtype.ProblemType.MultiClassClassificationProblem
import org.scalatest.{FunSuite, Matchers}

class SVMModelTest extends FunSuite with Matchers{

  ignore("testFitnessError") {

  }

  //TODO fix the test
  ignore("works for multiclass case (wine)") {
    val shufflingSeed = 1256

    val Array(trainDF, testDF) = Datasets.getWineDataframe(shufflingSeed).randomSplit(Array(0.8, 0.2))
    trainDF.cache()
    testDF.cache()

    val problemType = MultiClassClassificationProblem
    val svmF1 = SVMModel().fitnessError(trainDF, testDF, problemType).getCorrespondingMetric
    println(s"F1 computed for SVM model $svmF1")

    svmF1 shouldBe 0.9 +- 0.1
  }

  test("works for multiclass case ( glass )") {
    val shufflingSeed = 4567
    val Array(trainDF, testDF) = Datasets.getGlassDataFrame(shufflingSeed).randomSplit(Array(0.8, 0.2))
    trainDF.cache()
    testDF.cache()

    val problemType = MultiClassClassificationProblem
    val svmF1 = SVMModel().fitnessError(trainDF, testDF, problemType).getCorrespondingMetric
    println(s"F1 computed for SVM model $svmF1")

    svmF1 shouldBe 0.9 +- 0.1
  }

}

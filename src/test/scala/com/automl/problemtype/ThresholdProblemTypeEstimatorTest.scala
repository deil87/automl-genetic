package com.automl.problemtype

import com.automl.problemtype.ProblemType.{MultiClassClassificationProblem, RegressionProblem}
import com.automl.spark.SparkSessionProvider
import org.apache.spark.sql.DataFrame
import org.scalatest.{FunSuite, Matchers}
import utils.test.TestBase

class ThresholdProblemTypeEstimatorTest extends FunSuite with SparkSessionProvider with TestBase{
  import ss.implicits._

  val testFrame: DataFrame = ss.sparkContext.parallelize(
    Array(
      (2.2 , 10.8 , 3),
      (1.8 , 9.2 , 5),
      (2.1 , 10.6 , 6),
      (1.9 , 9.4 , 7)
    )
  ).toDF("param1", "param2", "label")

  test("estimate method should return Classification type of problem") {

    val estimatedProblemType: ProblemType = new ProblemTypeThresholdEstimator(5).estimate(testFrame, "label")
    estimatedProblemType.isInstanceOf[MultiClassClassificationProblem.type] should be(true)
  }

  test("estimate method should return Regression type of problem") {
    val estimatedProblemType: ProblemType = new ProblemTypeThresholdEstimator(3).estimate(testFrame, "label")
    estimatedProblemType.isInstanceOf[RegressionProblem.type] should be(true)
  }
}

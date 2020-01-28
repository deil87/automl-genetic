package com.automl.spark.linalg

import com.automl.spark.SparkSessionProvider
import org.apache.spark.mllib.linalg.Vectors
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpec}
import utils.test.TestBase

class SparkScaledAddSuite extends WordSpec with BeforeAndAfterAll with SparkSessionProvider with TestBase{


  val elementWiseAddition: (Array[Double], Array[Double]) => Array[Double] = { (x:Array[Double], y: Array[Double]) =>
    require(x.length == y.length)
    x.zip(y).map{ case(xn, yn) => xn + yn}
  }

  "Spark" should {

    "be able to add two vectors elementwisely" in {
      val numFeatures = 2

      var vectorOfParameters = Vectors.dense(Array.fill(numFeatures)(42.0))
      var vectorOfUpdates = Vectors.dense(Array.fill(numFeatures)(1.0))

      val customImplResult = elementWiseAddition(vectorOfParameters.toArray, vectorOfUpdates.toArray)

      customImplResult shouldBe Array(43.0, 43.0)

    }

    "be able to add two vectors elementwisely with linalg spark's function" in {
      val numFeatures = 2

      var vectorOfParameters = Vectors.dense(Array.fill(numFeatures)(42.0))
      var vectorOfUpdates = Vectors.dense(Array.fill(numFeatures)(1.0))

      breeze.linalg.axpy(1.0, vectorOfUpdates.toArray, vectorOfParameters.toArray)

      vectorOfParameters.toArray shouldBe Array(43.0, 43.0)

      breeze.linalg.axpy(-1.0, vectorOfUpdates.toArray, vectorOfParameters.toArray)

      vectorOfParameters.toArray shouldBe Array(42.0, 42.0)

    }
  }
}

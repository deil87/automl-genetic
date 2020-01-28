package com.automl.template.simple.perceptron

import com.automl.spark.SparkSessionProvider
import org.apache.spark.ml.linalg.SparseVector
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpec}
import com.test.TestBase


class LinearPerceptronWineClassificationSuite extends WordSpec with BeforeAndAfterAll with SparkSessionProvider with TestBase{

  ss.sparkContext.setLogLevel("ERROR")

  "LinearPerceptron" should {

    "be able to use OHE to represent target variable in a one-versus-others fashion" in {
      import org.apache.spark.ml.feature.OneHotEncoder

      val df = ss.createDataFrame(Seq(
        (0, 0),
        (1, 3),
        (2, 3),
        (3, 1),
        (4, 1),
        (5, 2)
      )).toDF("id", "category")

      val encoder = new OneHotEncoder().setDropLast(false)
        .setInputCol("category")
        .setOutputCol("categoryVec")

      val encoded = encoder.transform(df)
      val dense = encoded.select("categoryVec").collect().map(row => row.get(0).asInstanceOf[SparseVector].toDense)

      dense.slice(1,3).map(_.toArray(3)) shouldBe Array(1.0, 1.0)
      println(dense.map(_.toString()).mkString("\n"))
    }

  }

  override protected def afterAll(): Unit = {
//    ss.stop()
  }
}


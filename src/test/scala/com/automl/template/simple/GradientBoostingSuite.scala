package com.automl.template.simple

import com.automl.spark.SparkSessionProvider
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import org.scalatest.{Matchers, WordSpec}
import com.test.TestBase

class GradientBoostingSuite extends WordSpec with SparkSessionProvider with TestBase{
  import ss.implicits._

  "GradientBoodtingSuite" should {

    "should calculate fitness value" ignore {
      val estimator = GradientBoosting()

      val dataset: DataFrame = ss.sparkContext.parallelize(
        Array(
          (1 , 2 , 3),
          (3 , 4 , 5),
          (4 , 5 , 6),
          (5 , 6 , 7)
        )
      ).toDF("param1", "param2", "label")

//      dataset.show(10)

      val assembler = new VectorAssembler()
        .setInputCols(Array("param1", "param2"))
        .setOutputCol("features")

      val preparedTrainDS = assembler.transform(dataset)
//      preparedTrainDS.show(10)

      val testDS: DataFrame = ss.sparkContext.parallelize(
        Array(
          (4 , 5 , 6)
        )
      ).toDF("param1", "param2", "label")

      val preparedTestDS = assembler.transform(testDS)
//      preparedTestDS.show(10)

      val fitnessResult = estimator.fitnessError(preparedTrainDS, preparedTestDS)

//      fitnessResult.dfWithPredictions.show()

      fitnessResult.metricsMap shouldBe 0.0
    }
  }
}


package com.automl.template.simple

import com.automl.spark.SparkSessionProvider
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import org.scalatest.{Matchers, WordSpec}

class NeuralNetworkSuite extends WordSpec with Matchers with SparkSessionProvider {
  import ss.implicits._

  "NeuralNetwork" should {

    "should check that size of input layer correlates with number of features" in {

      // Note that MultilayerPerceptronClassifier will use numbers from `label` column
      // to estimate number of output classes (starting from 0 to max number in this column)
      val dataset: DataFrame = ss.sparkContext.parallelize(
        Array(
          (1 , 2 , 1),
          (3 , 4 , 2),
          (4 , 5 , 3),
          (5 , 6 , 4)
        )
      ).toDF("param1", "param2", "label")

      val assembler = new VectorAssembler()
        .setInputCols(Array("param1", "param2"))
        .setOutputCol("features")

      val preparedTrainDS = assembler.transform(dataset)

      assertThrows[IllegalArgumentException] {
        NeuralNetwork(Array(3, 3, 5)).fitnessError(preparedTrainDS, preparedTrainDS)
      }

      noException should be thrownBy NeuralNetwork(Array(2,3,5)).fitnessError(preparedTrainDS, preparedTrainDS)

    }
  }
}


package com.automl.template.simple.perceptron

import com.automl.classifier.LinearPerceptronClassifier
import com.automl.spark.SparkSessionProvider
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.scalatest.{Matchers, WordSpec}

class LinearPerceptronSuite extends WordSpec with Matchers with SparkSessionProvider {
  import ss.implicits._

  "LinearPerceptron" should {

    "be able to create matrix from features vector" in {

      val dataRDD: RDD[(Int, Int, Int)] = ss.sparkContext.parallelize(
        Array(
          (1, 2, 1),
          (3, 4, 2),
          (4, 5, 3),
          (5, 6, 4)
        )
      )

      val dataset: DataFrame = dataRDD.toDF("param1", "param2", "label")

      val assembler = new VectorAssembler()
        .setInputCols(Array("param1", "param2"))
        .setOutputCol("features")

      val preparedTrainDS = assembler.transform(dataset)

      val classifier = new LinearPerceptronClassifier()
      val matrix = classifier.extractFeaturesMatrix(preparedTrainDS.select("features"))

      val expectedMatrix = new IndexedRowMatrix(
        ss.sparkContext.parallelize(
          Array(
          IndexedRow(0 , Vectors.dense(1, 2)),
          IndexedRow(1 , Vectors.dense(3, 4)),
          IndexedRow(2 , Vectors.dense(4, 5)),
          IndexedRow(3 , Vectors.dense(5, 6))
          )
        )
      )
      matrix.rows.collect() shouldBe expectedMatrix.rows.collect()

    }

    "be able to train itself by perceptron learning algorithm" in {

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


    }
  }
}


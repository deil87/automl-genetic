package com.automl.template.simple.perceptron

import java.io.NotSerializableException

import breeze.linalg.DenseMatrix
import com.automl.classifier.LinearPerceptronClassifier
import com.automl.spark.SparkSessionProvider
import org.apache.spark.SparkException
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.scalatest.{Matchers, WordSpec}
import utils.LabeledVector

class LinearPerceptronSuite extends WordSpec with Matchers with SparkSessionProvider {
  import ss.implicits._

  def fixture =
    new {
      val dataRDD: RDD[(Int, Int, Int)] = ss.sparkContext.parallelize(
        Array(
          (4, 2, 1),
          (3, 4, 0),
          (4, 5, 0),
          (6, 3, 1)
        )
      )

      val dataset: DataFrame = dataRDD.toDF("param1", "param2", "label")

      val assembler = new VectorAssembler()
        .setInputCols(Array("param1", "param2"))
        .setOutputCol("features")

      val preparedTrainDS = assembler.transform(dataset)
    }


  import breeze.linalg.{DenseVector => BDV}

  def asBreeze(values: Array[Double]): BDV[Double] = new BDV[Double](values)

  "LinearPerceptron" should {

    "verify that DenseVector from breeze is mutable" in {

      val vectorOfParameters = Vectors.dense(Array.fill(5)(1.0))
      val vectorOfParametersBreeze = asBreeze(vectorOfParameters.toArray)

      val vectorToAdd = asBreeze(Array.fill(5)(2.0))

      vectorOfParametersBreeze += vectorToAdd

      vectorOfParametersBreeze.data shouldBe  Array.fill(5)(3.0)

    }

    "be able to calculate parameters through the iterative Perceptron learning algorithm( binary case)" ignore {
      val f = fixture
      val classifier = new LinearPerceptronClassifier()
      val activation = classifier.trainIterativelyMulticlasses(f.preparedTrainDS)
      val result = activation(0).toArray
      result(0) shouldBe 0.19033638814298248 +- 1.0
      result(1) shouldBe 3.8921764447475473  +- 0.8
      result(2) shouldBe -3.3293630571824764 +- 0.8

    }

    "be able to get number of classes for target variable" in {
      val f = fixture
      val classifier = new LinearPerceptronClassifier()
      val numberOfClasses = classifier.getNumberOfClasses(f.preparedTrainDS)
      numberOfClasses shouldBe 2

    }

    "be able to convert label column into one hot representation" in {
      val f = fixture
      val classifier = new LinearPerceptronClassifier()
      val withSeparateLabels = classifier.toOneHotRepresentationForTargetVariable(f.preparedTrainDS)
      withSeparateLabels.select("label_0").collect().slice(1, 3).map(_.get(0)) shouldBe Array(1.0, 1.0)
      withSeparateLabels.select("label_1").collect().filter(_.get(0) == 1.0).map(_.get(0)) shouldBe Array(1.0, 1.0)
    }

    "not be able to calculate elementwise addition with Breeze's vectors" ignore {
      val f = fixture
      import f.preparedTrainDS.sparkSession.implicits._

      def asBreeze(values: Array[Double]): BDV[Double] = new BDV[Double](values)

      def res = f.preparedTrainDS.as[LabeledVector].map{ row =>
        (asBreeze(row.features.toArray) + asBreeze(row.features.toArray)).toArray.sum
      }
      assertThrows[SparkException] {
        res.limit(1).collect()(0) shouldBe 42.0
      }
    }

    "be able to calculate elementwise addition with Breeze's vectors by collecting them locally" in {
      val f = fixture
      import f.preparedTrainDS.sparkSession.implicits._

      def asBreeze(values: Array[Double]): BDV[Double] = new BDV[Double](values)

      val res = f.preparedTrainDS.as[LabeledVector].collect().map{ row: LabeledVector =>
        (asBreeze(row.features.toArray) + asBreeze(row.features.toArray)).toArray.sum
      }

      res.mkString(",")
    }

    "not be able to calculate elementwise addition with Breeze's matrices" ignore {
      val f = fixture
      import f.preparedTrainDS.sparkSession.implicits._

      def asBreezeMatrix(values: Array[Double]): DenseMatrix[Double] = new DenseMatrix(1, 1, values)


      def res = f.preparedTrainDS.as[LabeledVector].map{ row =>
        (asBreezeMatrix(row.features.toArray) + asBreezeMatrix(row.features.toArray)).toArray.sum
      }
      assertThrows[SparkException] {
        res.limit(1).collect()(0) shouldBe 42.0
      }
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


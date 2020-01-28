package com.automl.spark

import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.DataFrame
import org.apache.spark.storage.StorageLevel
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpec}
import utils.TestBase

class StandardScalerSuite extends WordSpec with BeforeAndAfterAll with SparkSessionProvider with TestBase{

  import ss.implicits._
  import utils.SparkMLUtils._

  val initialDataset: DataFrame = ss.sparkContext.parallelize(
    Array(
      (2.2 , 10.8 , 3),
      (1.8 , 9.2 , 5),
      (2.1 , 10.6 , 6),
      (1.9 , 9.4 , 7)
    )
  ).toDF("param1", "param2", "label")

  case class Test(param1: Double, param2: Double, label: Int)

  initialDataset.persist(StorageLevel.MEMORY_ONLY)

  val features = Array("param1", "param2")

  val featuresAssembler = new VectorAssembler()
    .setInputCols(features)
    .setOutputCol("features")

  val withFeaturesAssembled = initialDataset.applyTransformation(featuresAssembler)

  "StandardScaler" should {

    "scale dataset columnwise" in {


      val scaler = new StandardScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
        .setWithStd(false)
        .setWithMean(true)

      val scalerModel = scaler.fit(withFeaturesAssembled)

      val scaledData = scalerModel.transform(withFeaturesAssembled)

      val scaledResult = scaledData.select("scaledFeatures").map(row => row.getAs[DenseVector](0).toArray(1)).toDF("param2")

      scaledResult.collect().forall(row => row.getAs[Double](0) < 1) shouldBe true

    }
  }
}

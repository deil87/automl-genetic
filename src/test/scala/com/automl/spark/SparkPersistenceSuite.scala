package com.automl.spark

import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.storage.StorageLevel
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpec}

class SparkPersistenceSuite extends WordSpec with Matchers with BeforeAndAfterAll with SparkSessionProvider {

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

  initialDataset.persist(StorageLevel.MEMORY_ONLY)

  val features = Array("param1", "param2")

  val featuresAssembler = new VectorAssembler()
    .setInputCols(features)
    .setOutputCol("features")

  val withFeaturesAssembled = initialDataset.applyTransformation(featuresAssembler)

  "Spark" should {

    "persist dataframes in a not transitive fashion" in {

      val scaler = new StandardScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
        .setWithStd(false)
        .setWithMean(true)


      // Compute summary statistics by fitting the StandardScaler.
      val scalerModel = scaler.fit(withFeaturesAssembled)

      // Normalize each feature to have unit standard deviation.
      val scaledData = scalerModel.transform(withFeaturesAssembled)

      initialDataset.storageLevel.useMemory shouldBe true

      withFeaturesAssembled.storageLevel.useMemory shouldBe false

      scaledData.storageLevel.useMemory shouldBe false

      println(ss.sparkContext.getPersistentRDDs.map(_._2.name))
      println("getRDDStorageInfo(also only cached rdd's):\n" + ss.sparkContext.getRDDStorageInfo.map(_.name).mkString(";"))

    }
  }
}

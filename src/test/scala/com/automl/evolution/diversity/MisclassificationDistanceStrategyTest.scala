package com.automl.evolution.diversity

import com.automl.spark.SparkSessionProvider
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.DataFrame
import org.scalatest.{BeforeAndAfterAll, FunSuite, Matchers}
import com.test.TestBase

class MisclassificationDistanceStrategyTest extends FunSuite with BeforeAndAfterAll with SparkSessionProvider with TestBase{

  import ss.implicits._
  import utils.SparkMLUtils._

  val df1: DataFrame = ss.sparkContext.parallelize(
    Array(
      (1 , 2 , 1),
      (2 , 3 , 4),
      (3 , 2 , 2),
      (4 , 3 , 3),
      (5 , 5 , 5),
      (6 , 1 , 1)
    )
  ).toDF("uniqueIdColumn", "prediction", "indexedLabel")

  val df2: DataFrame = ss.sparkContext.parallelize(
    Array(
      (1 , 2 , 1),
      (2 , 5 , 4),
      (3 , 2 , 2),
      (4 , 3 , 3),
      (5 , 5 , 5),
      (6 , 1 , 1)
    )
  ).toDF("uniqueIdColumn", "prediction", "indexedLabel")

  ignore("testGetDistance should return almost 1 for identical columns") {
    val misclassificationDistanceResult = new MisclassificationDistance().getDistance(df1, df2)
    misclassificationDistanceResult.distance should be(0.0 +- 0.001)

    misclassificationDistanceResult.errorsOnSamePositionsCount should be(2)
    misclassificationDistanceResult.matchesOnSamePositionsCount should be(4)

    misclassificationDistanceResult.rawData.select("similarMistakesDist").map(_.getDouble(0)).collect() should contain theSameElementsInOrderAs
      Seq( -1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    misclassificationDistanceResult.rawData.select("nonSimilarMistakesDist").map(_.getAs[DenseVector](0).values).collect() should contain theSameElementsInOrderAs
      Seq( Array(0, 0), Array(0, 0), Array(0, 0), Array(0, 0), Array(0, 0), Array(1.0, -1.0))
  }

}

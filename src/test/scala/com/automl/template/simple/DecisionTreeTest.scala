package com.automl.template.simple

import com.automl.spark.SparkSessionProvider
import org.apache.spark.ml.feature.StringIndexer
import org.scalatest.{FunSuite, Matchers}

class DecisionTreeTest extends FunSuite with SparkSessionProvider with Matchers{

  ignore("testFitnessError") {
    //TODO
  }

  ignore("test for Binary case") {
    //TODO
  }

  test("string indexes generates consistent indexes") {

    import ss.implicits._
    val threeDistinctLevelsDF = ss.sparkContext.parallelize(
      Seq(
        (1, "level1"),
        (2, "level1"),
        (3, "level2"),
        (4, "level3")
      )
    ).toDF("a", "label")

    val threeDistinctLevelsDFInDiffOrder = ss.sparkContext.parallelize(
      Seq(
        (1, "level3"),
        (2, "level2"),
        (3, "level2"),
        (4, "level1")
      )
    ).toDF("a", "label")

    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")

    val labelIndexer1 = indexer
      .fit(threeDistinctLevelsDF.sort("label"))

    val labelIndexer2 = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(threeDistinctLevelsDFInDiffOrder.sort("label"))

    println("Indexes:" + labelIndexer1.labels.mkString(" | "))
    println("Indexes:" + labelIndexer2.labels.mkString(" | "))

    labelIndexer1.labels should not contain theSameElementsInOrderAs(labelIndexer2.labels)
  }

}

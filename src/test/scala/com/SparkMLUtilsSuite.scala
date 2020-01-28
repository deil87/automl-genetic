package com

import com.automl.spark.SparkSessionProvider
import org.scalatest.{FunSuite, Matchers}
import utils.SparkMLUtils._
import utils.TestBase

class SparkMLUtilsSuite extends FunSuite with SparkSessionProvider with TestBase{

  import ss.implicits._

  val observations = ss.sparkContext.parallelize(
    Seq(
      ("2015-07-01T00:00:00.000+03:00", "10", 120.0),
      ("2015-07-02T00:00:00.000+03:00", "20", 130.0),
      ("2015-07-03T00:00:00.000+03:00", "NA", 140.0),
      ("2015-07-02T00:00:00.000+03:00", "NA", 150.0),
      ("2015-07-04T00:00:00.000+03:00", "NA", 160.0),
      ("2015-07-05T00:00:00.000+03:00", "NA", 170.0),
      ("2015-07-06T00:00:00.000+03:00", "50", 180.0)
    )
  ).toDF("timestamp", "num", "b")


  test("Calculate window function - return average over 3 previous") {

    val repaired = observations
      .toDouble("num")
      .repairNullWithRangeMean("num", -3)

//    repaired.showAll()

    2.0 should be(2.0)
  }

  test("toTrainTestPairs works") {

    val pairs = observations.toTrainTestPairs(3, 1234)

    pairs.foreach(pair => {
      pair._1.showAll()
      pair._2.showAll()
    })
  }

  test("Calculate window function - repairNullWithRangeMeanRate") {

    val repaired = observations
      .toDouble("num")
      .repairNullWithRangeMeanRate("num", -3)

//    repaired.showAll()

    repaired.filter($"b" === 140.0).select("num").first().getDouble(0) should be(20.0)
  }

  test("Calculate window function - should repair null's at the beginning") {

    val nullAtFirstRowBoservations = ss.sparkContext.parallelize(
      Seq(
        ("2015-07-01T00:00:00.000+03:00", "NA", 120.0),
        ("2015-07-02T00:00:00.000+03:00", "NA", 130.0),
        ("2015-07-03T00:00:00.000+03:00", "30", 140.0),
        ("2015-07-02T00:00:00.000+03:00", "40", 150.0),
        ("2015-07-06T00:00:00.000+03:00", "50", 180.0)
      )
    ).toDF("timestamp", "num", "b")

    val repaired = nullAtFirstRowBoservations
      .toDouble("num")
      .repairNullWithRangeMeanRate("num", -3)

//    repaired.showAll()

    repaired.filter($"b" === 140.0).select("num").first().getDouble(0) should be(30.0)
  }


}


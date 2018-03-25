package com.automl.spark.util

import com.automl.spark.SparkSessionProvider
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.rand
import org.scalatest.{FunSuite, Matchers}
import utils.SparkMLUtils

class PreparingDatasetsSuite extends FunSuite with Matchers with SparkSessionProvider{


  lazy val airlineDF: DataFrame = SparkMLUtils.loadResourceDF("/airline2008.csv")
    .select("DayOfWeek", "Distance", "DepTime", "CRSDepTime", "DepDelay")


  test("We sample from our dataset randomly") {

    airlineDF.orderBy(rand()).limit(100000)
      .coalesce(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .save("airline_sampled")
  }

}


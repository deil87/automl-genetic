package com.automl.spark.util

import com.automl.spark.SparkSessionProvider
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.rand
import org.scalatest.{FunSuite, Matchers}
import utils.{BenchmarkHelper, SparkMLUtils}

class PreparingDatasetsSuite extends FunSuite with Matchers with SparkSessionProvider{


  lazy val airlineDF: DataFrame = SparkMLUtils.loadResourceDF("/airline2008.csv")

  test("We sample from our dataset randomly") {

    airlineDF.orderBy(rand()).limit(100000)
      .coalesce(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .save("airline_sampled")
  }

  test("Convert airlineDF into parquet file with partitioning") {

    airlineDF.orderBy(rand()).limit(100000)
      .coalesce(1)
      .write
      .partitionBy("DayOfWeek")
      .parquet("airline_sampled.parquet")
  }

  test("Benchmark: Loading with Parquet is way more faster in case we need only couple of columns") {
    import ss.implicits._
    BenchmarkHelper.time("Loading csv") {
      val airlineDF = SparkMLUtils.loadResourceDF("/airline_allcolumns_sampled_100k.csv")
        .select("DayOfWeek", "Distance", "DepTime", "CRSDepTime", "DepDelay").where($"DayOfWeek" === 2)
      airlineDF.count()
    }

    BenchmarkHelper.time("Loading parquet") {
      val airlineDF = SparkMLUtils.loadParquet("src/test/resources/airline_allcolumns_sampled_100k_parquet")
        .select("DayOfWeek", "Distance", "DepTime", "CRSDepTime", "DepDelay").where($"DayOfWeek" === 2)
      airlineDF.count()
    }
  }

}


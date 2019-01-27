package com.automl.spark.util

import java.io.File
import java.nio.file.{Files, Paths}

import com.automl.spark.SparkSessionProvider
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.rand
import org.scalatest.{BeforeAndAfterAll, FunSuite, Matchers}
import utils.{BenchmarkHelper, SparkMLUtils}

class CsvVsParquetBenchmarkSuite extends FunSuite with Matchers with SparkSessionProvider with BeforeAndAfterAll{

  //TODO later move it to benchmarking subproject
  test("Benchmark: Loading with Parquet is way more faster in case we need only couple of columns") {

    lazy val airlineDFOriginal: DataFrame = SparkMLUtils.loadResourceDF("/airline2008_100mb.csv")
    //We sample from our dataset randomly
    airlineDFOriginal.orderBy(rand()).limit(100000)
      .coalesce(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .save("airline_sampled")

    //Convert airlineDF into parquet file with partitioning
    airlineDFOriginal.orderBy(rand()).limit(100000)
      .coalesce(1)
      .write
      .partitionBy("DayOfWeek")
      .parquet("airline_sampled.parquet")

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

    deleteRecursively(new File("airline_sampled.parquet"))
    deleteRecursively(new File("airline_sampled"))
  }

  def deleteRecursively(file: File): Unit = {
    if (file.isDirectory)
      file.listFiles.foreach(deleteRecursively)
    if (file.exists && !file.delete)
      throw new Exception(s"Unable to delete ${file.getAbsolutePath}")
  }

}


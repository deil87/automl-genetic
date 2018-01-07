package com.automl.spark

import org.apache.spark.sql.SparkSession

trait SparkSessionProvider {

    implicit val ss: SparkSession = SparkSessionProvider.ss

}

object SparkSessionProvider {

  implicit val ss: SparkSession = SparkSession.builder()
    .master("local[*]")

    .appName("AutoML application:")
    .config("spark.driver.memory", "6g")
    .config("spark.executor.memory", "6g")
    .getOrCreate()

  ss.sparkContext.setLogLevel("ERROR")
}

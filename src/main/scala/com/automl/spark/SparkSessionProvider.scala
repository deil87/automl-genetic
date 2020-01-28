package com.automl.spark

import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.sql.SparkSession

trait SparkSessionProvider extends LazyLogging{

    implicit val ss: SparkSession = SparkSessionProvider.ss

  logger.trace("Numbers of cached RDDs:"  + ss.sparkContext.getPersistentRDDs.size)

}

object SparkSessionProvider {

  implicit val ss: SparkSession = SparkSession.builder()
    .master("local[*]")

    .appName("AutoML application:")
    .config("spark.driver.memory", "6g")
    .config("spark.executor.memory", "6g")
    .getOrCreate()

  ss.sqlContext.clearCache()

//  for ((k,v) <- ss.sparkContext.getPersistentRDDs) {
//    v.unpersist()
//  }

  ss.sparkContext.setLogLevel("ERROR")
}

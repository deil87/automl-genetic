package com.automl.dataset

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}

trait SamplingStrategy {
  def sampleRatio(df: DataFrame, ratio: Double, seed: Long, byColumn: String = "indexedLabel"): DataFrame
  def sampleExactSize(df: DataFrame, ratio: Double, seed: Long, byColumn: String = "indexedLabel"): DataFrame
}

class RandomSampling extends SamplingStrategy {
  override def sampleExactSize(df: DataFrame, sampleSize: Double, seed: Long, byColumn: String = "indexedLabel"): Dataset[Row] = {
    import org.apache.spark.sql.functions.rand
    df.orderBy(rand(seed)).limit(sampleSize.toInt) // TODO Do we need to shuffle one more time here?
  }

  // Behaves as `sampleExact`
  override def sampleRatio(df: DataFrame, ratio: Double, seed: Long, byColumn: String): DataFrame = {
    val sampleSize = df.count() * ratio
    sampleExactSize(df, sampleSize, seed, byColumn)
  }
}

class StratifiedSampling extends SamplingStrategy {
  import utils.SparkMLUtils._
  override def sampleExactSize(df: DataFrame, ratio: Double, seed: Long, byColumn: String = "indexedLabel"): Dataset[Row] = {
    import df.sparkSession.implicits._
    import org.apache.spark.rdd.PairRDDFunctions
    import org.apache.spark.sql.functions._

    val data = df.rdd.keyBy(_.getAs[Double](byColumn))

    val fractions = data.map(_._1)
      .distinct
      .map(x => (x, ratio))
      .collectAsMap

    val sampleData = data
      .sampleByKeyExact(withReplacement = false, fractions, seed) // <-- here is the difference between sample and sampleExact
      .values


    df.sparkSession.createDataFrame(sampleData, df.schema)
  }

  override def sampleRatio(df: DataFrame, ratio: Double, seed: Long, byColumn: String = "indexedLabel"): Dataset[Row] = {
    import df.sparkSession.implicits._
    import org.apache.spark.rdd.PairRDDFunctions
    import org.apache.spark.sql.functions._

    val data = df.rdd.keyBy(_.getAs[Double](byColumn))

    val fractions = data.map(_._1)
      .distinct
      .map(x => (x, ratio))
      .collectAsMap

    val sampleData = data
      .sampleByKey(withReplacement = false, fractions, seed)
      .values


    df.sparkSession.createDataFrame(sampleData, df.schema)
  }
}

package com.automl.dataset

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.immutable
import scala.util.Random

trait SamplingStrategy {
  def sampleRatio(df: DataFrame, ratio: Double, seed: Long, byColumn: String = "indexedLabel"): DataFrame
  def sampleExactSize(df: DataFrame, sampleSize: Long, seed: Long, byColumn: String = "indexedLabel"): DataFrame
}

trait RowsSamplingStrategy extends SamplingStrategy
trait ColumnsSamplingStrategy extends SamplingStrategy


class RandomRowsSampling extends RowsSamplingStrategy {
  override def sampleExactSize(df: DataFrame, sampleSize: Long, seed: Long, byColumn: String = "indexedLabel"): Dataset[Row] = {
    import org.apache.spark.sql.functions.rand
    df.orderBy(rand(seed)).limit(sampleSize.toInt) // TODO Do we need to shuffle one more time here?
  }

  // Behaves as `sampleExact`
  override def sampleRatio(df: DataFrame, ratio: Double, seed: Long, byColumn: String): DataFrame = {
    val sampleSize = (df.count() * ratio).toLong
    sampleExactSize(df, sampleSize, seed, byColumn)
  }
}

class StratifiedRowsSampling extends RowsSamplingStrategy {
  import utils.SparkMLUtils._
  override def sampleExactSize(df: DataFrame, sampleSize: Long, seed: Long, byColumn: String = "indexedLabel"): Dataset[Row] = {
    import df.sparkSession.implicits._
    import org.apache.spark.rdd.PairRDDFunctions
    import org.apache.spark.sql.functions._

    val data = df.rdd.keyBy(_.getAs[Double](byColumn))

    require(sampleSize <= Double.MaxValue)

    val fractions = data.map(_._1)
      .distinct
      .map(x => (x, sampleSize.toDouble))
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

class RandomColumnsSampling extends ColumnsSamplingStrategy {
  override def sampleRatio(df: DataFrame, ratio: Double, seed: Long, byColumn: String): DataFrame = {
    val originalColumns = df.columns.toList
    val numberOfColumnsToSelect:Long = (originalColumns.size * ratio).toLong
    sampleExactSize(df, numberOfColumnsToSelect, seed, null)

  }

  override def sampleExactSize(df: DataFrame, sampleSize: Long, seed: Long, byColumn: String): DataFrame = {
    val originalColumns = df.columns.toList
    val selectedColumns: immutable.Seq[String] = new Random(seed).shuffle(originalColumns).take(sampleSize.toInt) // We do not support Long number of columns cases
    df.select(selectedColumns.head, selectedColumns.tail:_*)
  }
}

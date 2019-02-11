package com.automl.dataset

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}

trait SamplingStrategy {
  def sample(df: DataFrame, size: Long): DataFrame
}

class RandomSampling extends SamplingStrategy {
  override def sample(df: DataFrame, size: Long): Dataset[Row] = {
    import org.apache.spark.sql.functions.rand
    df.orderBy(rand()).limit(size.toInt)
  }
}

class StratifiedSampling extends SamplingStrategy {
  import utils.SparkMLUtils._
  override def sample(df: DataFrame, size: Long): Dataset[Row] = {
    import df.sparkSession.implicits._
    import org.apache.spark.rdd.PairRDDFunctions
    val data = df.rdd.keyBy(_.getAs[Double]("indexedLabel"))

    val fractions = data.map(_._1)
      .distinct
      .map(x => (x, 0.8))
      .collectAsMap

    /*val sampleData = data
      .sampleByKeyExact(withReplacement = false, fractions, 2L)
      .values*/

//    val (startValues,counts) = df.select("indexedLabel").rdd.map(value => value.getDouble(0)).histogram(6)

    val sampled = df.stat.sampleBy("indexedLabel", fractions.toMap, 1234L)

//    val (startValuesS,countsS) = sampled.select("indexedLabel").rdd.map(value => value.getDouble(0)).histogram(6)

//    sampled.showAllAndContinue
//    sampled.showCount_AndContinue("Sampled")
    sampled
  }
}

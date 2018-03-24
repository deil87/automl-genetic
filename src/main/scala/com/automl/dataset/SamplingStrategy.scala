package com.automl.dataset

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
  override def sample(df: DataFrame, size: Long): Dataset[Row] = {
    ???
  }
}

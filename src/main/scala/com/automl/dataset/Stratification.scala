package com.automl.dataset

import org.apache.spark.sql.{DataFrame, Dataset, Row}

class Stratification {
  def split(df: DataFrame, ratio: Double, seed: Long, byColumn: String = "indexedLabel"): Array[Dataset[Row]] = {

    val trainingSplit = new StratifiedSampling().sampleRatio(df, ratio,seed = seed)
    val testSplit = df.except(trainingSplit)
    Array(trainingSplit, testSplit)
  }
}

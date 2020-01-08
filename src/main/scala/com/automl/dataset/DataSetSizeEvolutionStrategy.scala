package com.automl.dataset

import org.apache.spark.sql.DataFrame

trait DataSetSizeEvolutionStrategy {
  def evolve(currentDF: DataFrame, newSize: Long, maxNumberOfEvolutions: Int, wholeDF: DataFrame, seed: Long)
            (implicit samplingStrategy: SamplingStrategy): DataFrame
}

class RandomDataSetSizeEvolutionStrategy extends DataSetSizeEvolutionStrategy {
  override def evolve(currentDF: DataFrame, newSize: Long, maxNumberOfEvolutions: Int, wholeDF: DataFrame, seed: Long)
                     (implicit samplingStrategy: SamplingStrategy): DataFrame = {
    val totalDataSize = wholeDF.count()

    val evolveToThisSize = if (newSize >= totalDataSize) totalDataSize else newSize
    val ratio = evolveToThisSize.toDouble / totalDataSize
    samplingStrategy.sampleRatio(wholeDF, ratio, seed)
  }
}

class AppendDataSetSizeEvolutionStrategy extends DataSetSizeEvolutionStrategy {

  override def evolve(currentDF: DataFrame, newSize: Long,  maxNumberOfEvolutions: Int, wholeDF: DataFrame, seed: Long)
                     (implicit samplingStrategy: SamplingStrategy):DataFrame = {
    val diffDF = wholeDF.except(currentDF)
    val currentDataSetSize = currentDF.count()
    val toBeAppendedDF = samplingStrategy.sampleExactSize(diffDF, newSize - currentDataSetSize, seed)
    currentDF.union(toBeAppendedDF)
  }

}


package com.automl.dataset

import org.apache.spark.sql.DataFrame

trait DataSetSizeEvolutionStrategy {
  def evolve(currentDF: DataFrame, newSize: Long, maxNumberOfEvolutions: Int, wholeDF: DataFrame)
            (implicit samplingStrategy: SamplingStrategy): DataFrame
}

class RandomDataSetSizeEvolutionStrategy extends DataSetSizeEvolutionStrategy {
  override def evolve(currentDF: DataFrame, newSize: Long, maxNumberOfEvolutions: Int, wholeDF: DataFrame)
                     (implicit samplingStrategy: SamplingStrategy): DataFrame = {
    val totalDataSize = wholeDF.count()
//    lazy val evolutionDataSizeFactor: Long = Math.max(totalDataSize / maxNumberOfEvolutions, 500)
    val evolveToThisSize = if (newSize >= totalDataSize) totalDataSize else newSize
    samplingStrategy.sample(wholeDF, evolveToThisSize)
  }
}

class AppendDataSetSizeEvolutionStrategy extends DataSetSizeEvolutionStrategy {

  override def evolve(currentDF: DataFrame, newSize: Long,  maxNumberOfEvolutions: Int, wholeDF: DataFrame)
                     (implicit samplingStrategy: SamplingStrategy):DataFrame = {
    val diffDF = wholeDF.except(currentDF)
    val currentDataSetSize = currentDF.count()
    val toBeAppendedDF = samplingStrategy.sample(diffDF, newSize - currentDataSetSize)
    currentDF.union(toBeAppendedDF)
  }

}


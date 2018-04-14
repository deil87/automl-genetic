package com.automl.dataset

import com.automl.spark.SparkSessionProvider
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.scalatest.{Matchers, WordSpec}
import utils.SparkMLUtils

class DataSetSizeEvolutionStrategyTest extends WordSpec  with Matchers with SparkSessionProvider{

  val total = 1000

  val testDF: DataFrame = SparkMLUtils.loadParquet("src/test/resources/airline_allcolumns_sampled_100k_parquet")
    .limit(total)
    .select("DayOfWeek", "Distance", "DepTime", "CRSDepTime", "DepDelay")
    .withColumn("uniqueIdColumn", monotonically_increasing_id)

  "Dataset size should" should {

    "evolve with AppendDataSetSizeEvolutionStrategy preserving previous rows" in {
      val currentSubSet = testDF.sample(false, 0.2)
      val currentSize: Long = currentSubSet.count()
      val nextSize = total / 2

      val anySamplingStrategy = new RandomSampling()

      val evolutionStrategy = new AppendDataSetSizeEvolutionStrategy()
      val evolvedDF =  evolutionStrategy.evolve(currentSubSet, nextSize, 5, testDF)(anySamplingStrategy)

      evolvedDF.cache()

      evolvedDF.count() should be(500)

      evolvedDF.intersect(currentSubSet).count() should be(currentSize)
    }

    "evolve with RandomDataSetSizeEvolutionStrategy preserving previous rows" in {
      val currentSubSet = testDF.sample(false, 0.2)
      val currentSize: Long = currentSubSet.count()
      val nextSize = total / 2

      val anySamplingStrategy = new RandomSampling()

      val evolutionStrategy = new RandomDataSetSizeEvolutionStrategy()
      val evolvedDF =  evolutionStrategy.evolve(currentSubSet, nextSize, 5, testDF)(anySamplingStrategy)

      evolvedDF.cache()

      evolvedDF.count() should be(500)

      val numberOfRowsFromPreviousEvolution = evolvedDF.intersect(currentSubSet).count()
      println(s"Number of rows from previous evolution: $numberOfRowsFromPreviousEvolution")

      // Probability is pretty low that all 500 randomly chosen will match with previous rows
      numberOfRowsFromPreviousEvolution should not be currentSize
    }

  }
}

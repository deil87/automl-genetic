package com.automl.dataset

import com.automl.spark.SparkSessionProvider
import org.apache.spark
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.scalatest.concurrent.TimeLimitedTests
import org.scalatest.time.{Minutes, Span}
import org.scalatest.{Matchers, WordSpec}
import utils.SparkMLUtils
import com.test.TestBase

import scala.util.Random

class DataSetSizeEvolutionStrategyTest extends WordSpec with SparkSessionProvider with TestBase {

  val total = 1000

  val testDF: DataFrame = SparkMLUtils.loadParquet("src/test/resources/airline_allcolumns_sampled_100k_parquet")
    .limit(total)
    .select("DayOfWeek", "Distance", "DepTime", "CRSDepTime", "DepDelay")
    .withColumn("uniqueIdColumn", monotonically_increasing_id)

  "Dataset size should" should {

    "evolve with AppendDataSetSizeEvolutionStrategy and preserving previous rows even if seeds are different" in {
      val seed = new Random().nextLong()
      val currentSubSet = testDF.sample(false, 0.2)
      val currentSize: Long = currentSubSet.count()
      val nextSize = total / 2

      val anySamplingStrategy = new RandomSampling()

      val evolutionStrategy = new AppendDataSetSizeEvolutionStrategy()
      val evolvedDF =  evolutionStrategy.evolve(currentSubSet, nextSize, 5, testDF, seed = seed + 1)(anySamplingStrategy)

      evolvedDF.cache()

      evolvedDF.count() should be(500)

      evolvedDF.intersect(currentSubSet).count() should be(currentSize)
    }

    "evolve with RandomDataSetSizeEvolutionStrategy without preserving previous rows if seeds are different" in {
      val seed = new Random().nextLong()

      val currentSubSet = testDF.sample(false, 0.1, seed)
//      currentSubSet.show(100, false)
      val currentSize: Long = currentSubSet.count()
      val nextSize = total / 2

      val anySamplingStrategy = new RandomSampling()

      val evolutionStrategy = new RandomDataSetSizeEvolutionStrategy()
      val evolvedDF =  evolutionStrategy.evolve(currentSubSet, nextSize, 5, testDF, seed = seed + 1)(anySamplingStrategy)

      evolvedDF.cache()

      evolvedDF.count() should be(500)

      val intersection = evolvedDF.intersect(currentSubSet)
//      intersection.show(100, false)

      val numberOfRowsFromPreviousEvolution = intersection.count()

      println(s"Number of rows from previous evolution: $numberOfRowsFromPreviousEvolution")

      // Probability is pretty low that new 500 randomly chosen rows will include all rows from previous evolution
      numberOfRowsFromPreviousEvolution should not be currentSize
    }

    "intersect is working" in {
      import ss.implicits._

      val datasetA: DataFrame = ss.sparkContext.parallelize(
        Array(
          (2.2 , 10.8 , 3),
          (1.8 , 9.2 , 5),
          (2.1 , 10.6 , 6),
          (1.9 , 9.4 , 7)
        )
      ).toDF("param1", "param2", "label")

      val datasetB: DataFrame = ss.sparkContext.parallelize(
        Array(
          (2.2 , 10.8 , 3)
        )
      ).toDF("param1", "param2", "differentLabel")  // regardless of the labels we compare columns anyway just by order

      datasetA.intersect(datasetB).count() shouldBe 1L
    }

  }
}

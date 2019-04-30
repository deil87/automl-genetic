package com.automl.dataset

import com.automl.spark.SparkSessionProvider
import org.scalatest.{FunSuite, Matchers, Retries}

import scala.util.Random

class StratifiedSamplingTest extends FunSuite with Matchers with SparkSessionProvider with Retries{

  test("Stratified should return ratio of the data per class") {
    val stratifier = new StratifiedSampling

    val seed = new Random().nextLong()

    val glassDF = Datasets.getGlassDataFrame(1234)
    val sampled = stratifier.sample(glassDF, 0.5, seed)

    val (startValues, counts) = glassDF.select("indexedLabel").rdd.map(value => value.getDouble(0)).histogram(6)
    val (startValuesS, countsS) = sampled.select("indexedLabel").rdd.map(value => value.getDouble(0)).histogram(6)

    counts(0).toDouble / 2 shouldBe countsS(0).toDouble +- 5
  }

  test("Stratified should preserve all levels") {
    val seed = new Random().nextLong()

    val stratifier = new StratifiedSampling
    import utils.SparkMLUtils._
    import ss.implicits._

    val observations = ss.sparkContext.parallelize(
      Seq(
        (0.0),
        (0.0),
        (0.0),
        (0.0),
        (1.0),
        (1.0),
        (2.0),
        (3.0),
        (3.0),
        (3.0)

      )
    ).toDF("grades")

    val sampled = stratifier.sample(observations, 0.5, seed, "grades").cache()

    sampled.showAllAndContinue

    sampled.filter($"grades" === 0.0).count().toDouble shouldBe 2.0
    sampled.filter($"grades" === 1.0).count().toDouble shouldBe 1.0
    sampled.filter($"grades" === 2.0).count() shouldBe 1
    sampled.filter($"grades" === 3.0).count() shouldBe 2
  }


}

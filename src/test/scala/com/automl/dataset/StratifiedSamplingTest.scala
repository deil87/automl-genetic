package com.automl.dataset

import com.automl.spark.SparkSessionProvider
import org.apache.spark.sql.DataFrame
import org.scalatest.{FunSuite, Matchers, Retries}

import scala.collection.mutable
import scala.util.Random

class StratifiedSamplingTest extends FunSuite with Matchers with SparkSessionProvider with Retries{

  test("Stratified should return ratio of the data per class") {
    val stratifier = new StratifiedSampling

    val seed = new Random().nextLong()

    val glassDF = Datasets.getGlassDataFrame(1234)
    val sampled = stratifier.sampleExact(glassDF, 0.5, seed)

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

    val sampled = stratifier.sampleExact(observations, 0.5, seed, "grades").cache()

//    sampled.showAllAndContinue

    sampled.filter($"grades" === 0.0).count().toDouble shouldBe 2.0
    sampled.filter($"grades" === 1.0).count().toDouble shouldBe 1.0
    sampled.filter($"grades" === 2.0).count() shouldBe 1
    sampled.filter($"grades" === 3.0).count() shouldBe 2
  }

  test("StratifiedSampling.sampleExact should sample different rows depending on seed by same number of them") {

    val stratifier = new StratifiedSampling

    val irisDF = Datasets.getIrisDataFrame(1234).cache()

    val samples = mutable.Buffer[DataFrame]()
    for(i <- 1 to 5) {
      val seed = new Random().nextLong()
      val sampled = stratifier.sampleExact(irisDF, 0.8, seed).cache()
      println("Size of the sample is: " + sampled.count())
      require(samples.forall(previousSample => previousSample.except(sampled).count() != 0), "Samples should be different")
      samples.append(sampled)
    }

    samples.map(_.count()).contains(samples.head.count()) shouldBe true
  }

  test("StratifiedSampling.sample should sample different rows depending on seed by same number of them") {

    val stratifier = new StratifiedSampling

    val irisDF = Datasets.getIrisDataFrame(1234).cache()

    val samples = mutable.Buffer[DataFrame]()
    for(i <- 1 to 5) {
      val seed = new Random().nextLong()
      val sampled = stratifier.sample(irisDF, 0.8, seed).cache()
      println("Size of the sample is: " + sampled.count())
      require(samples.forall(previousSample => previousSample.except(sampled).count() != 0), "Samples should be different")
      samples.append(sampled)
    }

    samples.map(_.count()).distinct.length >=2 shouldBe true
  }


}

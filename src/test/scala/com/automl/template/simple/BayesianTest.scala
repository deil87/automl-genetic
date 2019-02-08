package com.automl.template.simple

import com.automl.dataset.Datasets
import com.automl.problemtype.ProblemType
import com.automl.spark.SparkSessionProvider
import org.scalatest.{FunSuite, Matchers}

class BayesianTest extends FunSuite with SparkSessionProvider with Matchers{
  import utils.SparkMLUtils._


  val problemType = ProblemType.MultiClassClassificationProblem

  test("testFitnessError witn wine dataset") {
    val Array(trainDF, testDF) = Datasets.getWineDataframe.randomSplit(Array(0.8, 0.2))
    trainDF.cache()
    testDF.cache()

    val bayesianF1 = Bayesian().fitnessError(trainDF, testDF, problemType).getCorrespondingMetric
    println(s"F1 computed for Bayesian model $bayesianF1")

    bayesianF1 shouldBe 0.9 +- 0.1
  }

  test("testFitnessError with Glass dataset") {
    val seed = 1234
    val seedForShuffling = 3456
    // seed only makes sure the sizes of the splits are constant not the elements inside. seedForShuffling serves for tat purpose
    val Array(trainDF, testDF) = Datasets.getGlassDataFrame(seedForShuffling).randomSplit(Array(0.8, 0.2), seed)
    val Array(trainDF2, testDF2) = Datasets.getGlassDataFrame(seedForShuffling).randomSplit(Array(0.8, 0.2), seed)

    require(trainDF.count() == trainDF2.count(), "Sizes of the splits should be equal")

//    trainDF.describe("indexedLabel").show(100)
//    trainDF.printSchema()
    trainDF.cache()
    testDF.cache()

    //Uncomment to check label encodings
    //    trainDF.select("indexedLabel", "TypeOfGlass").showAllAndContinue

    val (startValues,counts) = trainDF.select("indexedLabel").rdd.map(value => value.getDouble(0)).histogram(6)
    val (startValuesTest,countsTest) = testDF.select("indexedLabel").rdd.map(value => value.getDouble(0)).histogram(6)

    val zippedValues = startValues.zip(counts)
    case class HistRow(startPoint:Double,count:Long)
    val rowRDD = zippedValues.map( value => HistRow(value._1,value._2))
    println(rowRDD.mkString(" , "))

    val bayesianF1 = Bayesian().fitnessError(trainDF.union(testDF), testDF, problemType).getCorrespondingMetric
    println(s"F1 computed for Bayesian model $bayesianF1")

    bayesianF1 shouldBe 0.9 +- 0.1
  }

}

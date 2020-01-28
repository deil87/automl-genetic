package com.automl.template.simple

import com.automl.ConfigProvider
import com.automl.dataset.Datasets
import com.automl.evolution.dimension.hparameter._
import com.automl.problemtype.ProblemType
import com.automl.spark.SparkSessionProvider
import com.typesafe.config.{Config, ConfigFactory}
import com.typesafe.scalalogging.LazyLogging
import org.scalatest.{FunSuite, Matchers}
import utils.TestBase

class BayesianTest extends FunSuite with SparkSessionProvider with TestBase{
  import utils.SparkMLUtils._

  val testOverride: Config = ConfigFactory.parseString(
    """
      |evolution {
      |  hpGridSearch = false
      |  hyperParameterDimension {
      |     enabled = false
      |  }
      |}
    """.stripMargin)
  ConfigProvider.clearOverride.addOverride(testOverride)

  val problemType = ProblemType.MultiClassClassificationProblem

  test("Bayesian model can predict for wine dataset") {

    ConfigProvider.clearOverride.addOverride(
      s"""
         |evolution {
         |  hyperParameterDimension {
         |    enabled = false
         |  }
         |  evaluation {
         |    multiclass.metric = "f1"
         |  }
         |}
        """)

    val shufflingSeed = 1256

    val Array(trainDF, testDF) = Datasets.getWineDataframe(shufflingSeed).randomSplit(Array(0.8, 0.2))
    trainDF.cache()
    testDF.cache()

    val bayesianHPFieldOpt = Some(HyperParametersField(
      Seq(
        BayesianHPGroup(Seq(Smoothing(Some(2.0))))
      )
    ))

    val bayesianF1 = Bayesian().fitnessError(trainDF, testDF, problemType, bayesianHPFieldOpt).getCorrespondingMetric
    println(s"F1 computed for Bayesian model $bayesianF1")

    bayesianF1 shouldBe 0.9 +- 0.1
  }


  test("Bayesian model might predict same values depending on the Smoothing hyper parameter's value and data splits") {

    ConfigProvider.clearOverride.addOverride(
      s"""
         |evolution {
         |  hyperParameterDimension {
         |    enabled = false
         |  }
         |  evaluation {
         |    multiclass.metric = "f1"
         |  }
         |}
        """)

    val seed = 1234
    val shufflingSeed = 1256

    val Array(trainDF, testDF) = Datasets.getWineDataframe(shufflingSeed).randomSplit(Array(0.8, 0.2), seed)
    trainDF.cache()
    testDF.cache()

    val bayesianHPFieldOpt_1 = Some(HyperParametersField(
      Seq(
        BayesianHPGroup(Seq(Smoothing(Some(2.0))))
      )
    ))

    val bayesianHPFieldOp_2 = Some(HyperParametersField(
      Seq(
        BayesianHPGroup(Seq(Smoothing(Some(3.0))))
      )
    ))

    val bayesianHPFieldOp_3 = Some(HyperParametersField(
      Seq(
        BayesianHPGroup(Seq(Smoothing(Some(8.0))))
      )
    ))

    val bayesianF1_1 = Bayesian().fitnessError(trainDF, testDF, problemType, bayesianHPFieldOpt_1).getCorrespondingMetric
    val bayesianF1_2 = Bayesian().fitnessError(trainDF, testDF, problemType, bayesianHPFieldOp_2).getCorrespondingMetric
    val bayesianF1_3 = Bayesian().fitnessError(trainDF, testDF, problemType, bayesianHPFieldOp_3).getCorrespondingMetric

    println(s"F1 computed for Bayesian model with Smoothing = 2 : $bayesianF1_1")
    println(s"F1 computed for Bayesian model with Smoothing = 3 : $bayesianF1_2")
    println(s"F1 computed for Bayesian model with Smoothing = 8 : $bayesianF1_3")

    bayesianF1_1 shouldBe bayesianF1_2
    bayesianF1_1 should not be bayesianF1_3
  }

  test("testFitnessError with Glass dataset") {
    ConfigProvider.addOverride(
      """
        |evolution {
        |  evaluation {
        |    multiclass.metric = "f1"
        |  }
        |}
      """)

    val seed = 1234
    val seedForShuffling = 3456
    // seed only makes sure the sizes of the splits are constant not the elements inside. seedForShuffling serves for tat purpose
    val Array(trainDF, testDF) = Datasets.getGlassDataFrame(seedForShuffling).randomSplit(Array(0.8, 0.2), seed)
    trainDF.cache()
    testDF.cache()

    val bayesianHPFieldOpt = Some(HyperParametersField(
      Seq(
        BayesianHPGroup(Seq(Smoothing(Some(7.0))))
      )
    ))

    //Uncomment to check label encodings
    //    trainDF.select("indexedLabel", "TypeOfGlass").showAllAndContinue

    val (startValues,counts) = trainDF.select("indexedLabel").rdd.map(value => value.getDouble(0)).histogram(6)
    val (startValuesTest,countsTest) = testDF.select("indexedLabel").rdd.map(value => value.getDouble(0)).histogram(6)

    val zippedValues = startValues.zip(counts)
    case class HistRow(startPoint:Double,count:Long)
    val rowRDD = zippedValues.map( value => HistRow(value._1,value._2))
    println(rowRDD.mkString(" , "))

    val bayesianF1 = Bayesian().fitnessError(trainDF, testDF, problemType, bayesianHPFieldOpt).getCorrespondingMetric
    println(s"F1 computed for Bayesian model $bayesianF1")

    bayesianF1 shouldBe 0.4 +- 0.1
  }

}

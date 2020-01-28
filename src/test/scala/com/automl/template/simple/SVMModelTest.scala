package com.automl.template.simple

import com.automl.ConfigProvider
import com.automl.dataset.Datasets
import com.automl.evolution.dimension.hparameter._
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.MultiClassClassificationProblem
import com.automl.template.LeafTemplate
import com.typesafe.config.{Config, ConfigFactory}
import org.scalatest.{FunSuite, Matchers}
import utils.TestBase

class SVMModelTest extends FunSuite with TestBase{

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


  ignore("testFitnessError") {

  }

  //TODO fix the test
  ignore("works for multiclass case (wine)") {
    val shufflingSeed = 1256

    val Array(trainDF, testDF) = Datasets.getWineDataframe(shufflingSeed).randomSplit(Array(0.8, 0.2))
    trainDF.cache()
    testDF.cache()

    val problemType = MultiClassClassificationProblem
    val svmF1 = SVMModel().fitnessError(trainDF, testDF, problemType).getCorrespondingMetric
    println(s"F1 computed for SVM model $svmF1")

    svmF1 shouldBe 0.9 +- 0.1
  }

  // OneVsRest is not yet working
  ignore("works for multiclass case ( glass )") {
    val shufflingSeed = 4567
    val Array(trainDF, testDF) = Datasets.getGlassDataFrame(shufflingSeed).randomSplit(Array(0.8, 0.2))
    trainDF.cache()
    testDF.cache()

    val svmHPFieldOpt = Some(HyperParametersField(
      Seq(
        SVMHPGroup(
          Seq(
            MaxIterSVM(Some(50.0)),
            RegParamSVM(Some(0.3)))
        )
      )
    ))

    val problemType = MultiClassClassificationProblem
    val svmF1 = SVMModel().fitnessError(trainDF, testDF, problemType, svmHPFieldOpt).getCorrespondingMetric
    println(s"F1 computed for SVM model $svmF1")

    svmF1 shouldBe 0.9 +- 0.1
  }


  // AG-184
  ignore("logloss metric is being computed successfully for SVMModel") {

    val metric = "logloss"

    ConfigProvider.addOverride(
      s"""
         |evolution {
         |  hyperParameterDimension {
         |    enabled = false
         |  }
         |  evaluation {
         |    multiclass.metric = "$metric"
         |  }
         |}
        """)

    val template = LeafTemplate(SVMModel())

    val seed = 1234
    val preparedGlassDF = Datasets.getGlassDataFrame(seed)

    val Array(trainDF, testDF) = preparedGlassDF.randomSplit(Array(0.8, 0.2))

    val result = template.evaluateFitness(trainDF, testDF, ProblemType.MultiClassClassificationProblem, None, seed)
    result.getCorrespondingMetric should be >= 0.0
  }

}

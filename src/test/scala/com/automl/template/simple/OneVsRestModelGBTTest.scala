package com.automl.template.simple

import com.automl.ConfigProvider
import com.automl.dataset.Datasets
import com.automl.problemtype.ProblemType
import com.automl.spark.SparkSessionProvider
import com.automl.template.LeafTemplate
import org.scalatest.{BeforeAndAfterAll, FunSuite, Matchers}
import utils.SparkMLUtils
import com.test.TestBase

class OneVsRestModelGBTTest extends FunSuite with BeforeAndAfterAll with SparkSessionProvider with TestBase{

  ss.sparkContext.setLogLevel("ERROR")
  import ss.implicits._
  import utils.SparkMLUtils._


  test("be able to separate dataset Cars into multiple classes( multiclass case) with LogisticRegression") {

    ConfigProvider.clearOverride.addOverride(
      s"""
         |evolution {
         |  hyperParameterDimension {
         |    enabled = false
         |  }
         |  evaluation {
         |    multiclass.metric = "logloss"
         |  }
         |}
        """)

    val seed = 1234
    val preparedCarDF = Datasets.getCarDataFrame(seed)

    val template = LeafTemplate(OneVsRestGBTModel())

    val Array(trainDF, testDF) = preparedCarDF.randomSplit(Array(0.8, 0.2))

    val result = template.evaluateFitness(trainDF, testDF, ProblemType.MultiClassClassificationProblem, None, seed)

    ClassificationMetricsHelper.printConfusionMatrix("ovr", result.dfWithPredictions, testDF)

    result.getCorrespondingMetric should be >= 0.0

  }


}

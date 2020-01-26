package com.automl.template.simple

import com.automl.ConfigProvider
import com.automl.dataset.Datasets
import com.automl.problemtype.ProblemType
import com.automl.spark.SparkSessionProvider
import com.automl.template.{LeafTemplate, TemplateTree}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.scalatest.{BeforeAndAfterAll, FunSuite, Matchers}
import utils.SparkMLUtils

class XGBoostEstimatorTest extends FunSuite with Matchers with BeforeAndAfterAll with SparkSessionProvider {

  import ss.implicits._
  import utils.SparkMLUtils._


  test("be able to separate dataset Cars into multiple classes( multiclass case) with XGBoostEstimator") {

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
    preparedCarDF.showN_AndContinue(10)

    val template = LeafTemplate(XGBoostEstimator())

    val Array(trainDF, testDF) = preparedCarDF.randomSplit(Array(0.8, 0.2))

    val result = template.evaluateFitness(trainDF, testDF, ProblemType.MultiClassClassificationProblem, None, seed)

    ClassificationMetricsHelper.printConfusionMatrix("xgboost", result.dfWithPredictions, testDF)

    result.getCorrespondingMetric should be >= 0.0

  }

}

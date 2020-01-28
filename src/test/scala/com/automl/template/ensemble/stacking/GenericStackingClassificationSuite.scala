package com.automl.template.ensemble.stacking

import com.automl.ConfigProvider
import com.automl.dataset.Datasets
import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.problemtype.ProblemType
import com.automl.spark.SparkSessionProvider
import com.automl.template.LeafTemplate
import com.automl.template.simple.{Bayesian, DecisionTree, LinearRegressionModel, LogisticRegressionModel}
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.sql.functions.{monotonically_increasing_id, rand}
import org.scalatest.{BeforeAndAfterAll, FunSuite, Matchers}
import utils.SparkMLUtils
import utils.test.TestBase

import scala.util.Random

class GenericStackingClassificationSuite extends FunSuite with BeforeAndAfterAll with SparkSessionProvider with TestBase{

  ss.sparkContext.setLogLevel("ERROR")
  import ss.implicits._
  import utils.SparkMLUtils._

  val shufflingSeed = 1256
  val Array(trainDF, testDF) = Datasets.getWineDataframe(shufflingSeed).randomSplit(Array(0.8, 0.2))
  trainDF.cache()
  testDF.cache()

  //TODO fix the test
  ignore("be able to separate  classes( multiclass case) with GenericStacking") {

    val genericStacking = GenericStacking(unusedMetaLearner = new GBTRegressor())

    val models = Seq(
      LeafTemplate(DecisionTree()),
      LeafTemplate(Bayesian()),
      LeafTemplate(LogisticRegressionModel())
    )

    val seed = new Random().nextLong()

    val problemType = ProblemType.MultiClassClassificationProblem
    val fitnessResult = genericStacking.ensemblingFitnessError(trainDF, testDF, models, problemType, hyperParamsField = Some(HyperParametersField.default), seed = seed)


    val gsf1 = fitnessResult.getCorrespondingMetric

    val decisionTreeF1 = DecisionTree().fitnessError(trainDF, testDF, problemType).getCorrespondingMetric
    println(s"F1 computed for DecisionTree model $decisionTreeF1")

    gsf1 should be >= decisionTreeF1
    gsf1 shouldBe 0.9 +- 0.1

  }

  // AG-201
  ignore("be able to separate dataset classes( multiclass case) with GenericStacking over Bayesian model") {

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

    val genericStacking = GenericStacking(unusedMetaLearner = new GBTRegressor())

    val models = Seq(
      LeafTemplate(Bayesian())
    )

    val problemType = ProblemType.MultiClassClassificationProblem

    val seed = new Random().nextLong()

    val fitnessResult = genericStacking.ensemblingFitnessError(trainDF, testDF, models, problemType, hyperParamsField = Some(HyperParametersField.default), seed = seed)


    val gsf1 = fitnessResult.getCorrespondingMetric

    gsf1 shouldBe 0.9 +- 0.1

  }

  // AG-201
  ignore("be able to separate dataset classes( multiclass case) with GenericStacking over DecisionTree models") {

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

    val genericStacking = GenericStacking(unusedMetaLearner = new GBTRegressor())

//    trainDF.showN_AndContinue(30, "Preprocessed Wine dataset")
    trainDF.describe("indexedLabel")
//      .showN_AndContinue(50, "Describe of the label column")
    val models = Seq(
      LeafTemplate(DecisionTree()),
      LeafTemplate(Bayesian())
    )

    val problemType = ProblemType.MultiClassClassificationProblem

    val seed = new Random().nextLong()

    val fitnessResult = genericStacking.ensemblingFitnessError(trainDF, testDF, models, problemType, hyperParamsField = Some(HyperParametersField.default), seed = seed)


    val gsf1 = fitnessResult.getCorrespondingMetric

    val decisionTreeF1 = DecisionTree().fitnessError(trainDF, testDF, problemType).getCorrespondingMetric
    println(s"F1 computed for DecisionTree model $decisionTreeF1")

    gsf1 should be >= decisionTreeF1

    gsf1 shouldBe 0.9 +- 0.1

  }
}

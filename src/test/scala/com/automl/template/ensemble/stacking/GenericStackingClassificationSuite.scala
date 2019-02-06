package com.automl.template.ensemble.stacking

import com.automl.dataset.Datasets
import com.automl.problemtype.ProblemType
import com.automl.spark.SparkSessionProvider
import com.automl.template.LeafTemplate
import com.automl.template.simple.{Bayesian, DecisionTree, LinearRegressionModel, LogisticRegressionModel}
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.sql.functions.{monotonically_increasing_id, rand}
import org.scalatest.{BeforeAndAfterAll, FunSuite, Matchers}
import utils.SparkMLUtils

class GenericStackingClassificationSuite extends FunSuite with Matchers with BeforeAndAfterAll with SparkSessionProvider {

  ss.sparkContext.setLogLevel("ERROR")
  import ss.implicits._
  import utils.SparkMLUtils._

  val Array(trainDF, testDF) = Datasets.getWineDataframe.randomSplit(Array(0.8, 0.2))
  trainDF.cache()
  testDF.cache()

  test("be able to separate  classes( multiclass case) with GenericStacking") {

    val genericStacking = GenericStacking(unusedMetaLearner = new GBTRegressor())

    val models = Seq(
      LeafTemplate(DecisionTree()),
      LeafTemplate(Bayesian()),
      LeafTemplate(LogisticRegressionModel())
    )

    val problemType = ProblemType.MultiClassClassificationProblem
    val fitnessResult = genericStacking.ensemblingFitnessError(trainDF, testDF, models, problemType)


    val gsf1 = fitnessResult.getCorrespondingMetric

    val decisionTreeF1 = DecisionTree().fitnessError(trainDF, testDF, problemType).getCorrespondingMetric
    println(s"F1 computed for DecisionTree model $decisionTreeF1")

    gsf1 should be >= decisionTreeF1
    gsf1 shouldBe 0.9 +- 0.1

  }

  test("be able to separate dataset classes( multiclass case) with GenericStacking over Bayesian model") {

    val genericStacking = GenericStacking(unusedMetaLearner = new GBTRegressor())

    val models = Seq(
      LeafTemplate(Bayesian())
    )

    val problemType = ProblemType.MultiClassClassificationProblem
    val fitnessResult = genericStacking.ensemblingFitnessError(trainDF, testDF, models, problemType)


    val gsf1 = fitnessResult.getCorrespondingMetric

    gsf1 shouldBe 0.9 +- 0.1

  }

  test("be able to separate dataset classes( multiclass case) with GenericStacking over DecisionTree models") {

    val genericStacking = GenericStacking(unusedMetaLearner = new GBTRegressor())

    val models = Seq(
      LeafTemplate(DecisionTree()),
      LeafTemplate(Bayesian())
    )

    val problemType = ProblemType.MultiClassClassificationProblem
    val fitnessResult = genericStacking.ensemblingFitnessError(trainDF, testDF, models, problemType)


    val gsf1 = fitnessResult.getCorrespondingMetric

    val decisionTreeF1 = DecisionTree().fitnessError(trainDF, testDF, problemType).getCorrespondingMetric
    println(s"F1 computed for DecisionTree model $decisionTreeF1")

    gsf1 should be >= decisionTreeF1

    gsf1 shouldBe 0.9 +- 0.1

  }
}
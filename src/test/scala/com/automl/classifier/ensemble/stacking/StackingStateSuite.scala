package com.automl.classifier.ensemble.stacking

import com.automl.spark.SparkSessionProvider
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{GBTRegressor, LinearRegression}
import org.apache.spark.sql.DataFrame
import org.scalatest.{FunSuite, Matchers}
import utils.SparkMLUtils


class StackingStateSuite extends FunSuite with Matchers with SparkSessionProvider{

  import utils.SparkMLUtils._

  import ss.implicits._

  ignore("Should apply models fold-wise and return dataframe with additional prediction columns") {
    val observations = ss.sparkContext.parallelize(
      Seq(
        (1, 1, 100.0),
        (2, 2, 200.0),
        (3, 1, 200.0),
        (4, 2, 300.0),
        (5, 1, 100.0),
        (6, 2, 200.0),
        (7, 1, 200.0),
        (8, 2, 300.0),
        (9, 1, 100.0),
        (10, 2, 200.0),
        (11, 1, 200.0),
        (12, 2, 300.0),
        (13, 1, 100.0),
        (14, 2, 200.0),
        (15, 1, 200.0),
        (16, 2, 300.0)
      )
    ).toDF("id", "num", "label")

    def featuresAssembler = {
      new VectorAssembler()
        .setInputCols(Array("num"))
        .setOutputCol("features")
    }

    val preparedObservations = featuresAssembler.transform(observations)

    val stackingStage = new Stacking(3)

    val predictor1 = new LinearRegression()
    val predictor2 = new GBTRegressor()

    val predictors = Array(predictor1, predictor2)
    stackingStage.setPredictors(predictors)

    val transformed = stackingStage.firstStage(preparedObservations)

    transformed.showAll()
    transformed.schema.fields.exists(_.name == "foldId") should be(true)
    transformed.schema.fields.exists(_.name == "prediction0") should be(true)
    transformed.schema.fields.exists(_.name == "prediction1") should be(true)

  }


  test("Should increase accuracy of prediction") {

    import breeze.linalg.{DenseVector => DenseVectorBreeze, _}
    import breeze.stats.distributions._

    val N = 1000
    val variance = 3.0
    val covariance = 0.0
    val meanMatrix = new DenseVectorBreeze(Array(50.0, 10.0))
    val covMatrix = new DenseMatrix[Double](2,2, Array(variance, covariance ,covariance, variance))
    val gaussian = new MultivariateGaussian(mean = meanMatrix, covMatrix)
    val sample: IndexedSeq[DenseVectorBreeze[Double]] = gaussian.sample(N)

    val samplesFirstVar = sample.map(_.data(0)).toArray
    val samplesSecondVar = sample.map(_.data(1)).toArray

    val observations: DataFrame = ss.sparkContext.parallelize(
      samplesFirstVar.zip(samplesSecondVar).zipWithIndex.map { case ((independent, dependent), idx) =>
        (idx, independent, dependent)
      }
    ).toDF("id", "num", "label")


    def featuresAssembler = {
      new VectorAssembler()
        .setInputCols(Array("num"))
        .setOutputCol("features")
    }

    val preparedObservations = featuresAssembler.transform(observations)

    val Array(trainingSplit,testSplit)  = preparedObservations.randomSplit(Array(0.8, 0.2),11L)

    val stacking = new Stacking(3)

    val predictor1 = new LinearRegression()
    val predictor3 = new LinearRegression().setFitIntercept(false).setRegParam(0.1)
    val predictor2 = new GBTRegressor()

    stacking.setPredictors(Array(predictor1, predictor2, predictor3))
    stacking.setMetaPredictor(predictor1)

    val transformedTraining = stacking.firstStage(trainingSplit).cache()

    val evaluator = new RegressionEvaluator()

    println("RMSE 0:" + evaluator.evaluate(transformedTraining.withColumnRenamed("prediction0", "prediction")))
    println("RMSE 1:" + evaluator.evaluate(transformedTraining.withColumnRenamed("prediction1", "prediction")))
    println("RMSE 2:" + evaluator.evaluate(transformedTraining.withColumnRenamed("prediction2", "prediction")))

    transformedTraining.showAll()

    val transformedTest = stacking.secondStage(trainingSplit, testSplit)

    val finalPredictions = stacking.finalStage(transformedTraining, transformedTest).cache()

    finalPredictions.showAll()

    println("RMSE Final:" + evaluator.evaluate(finalPredictions))
  }

}


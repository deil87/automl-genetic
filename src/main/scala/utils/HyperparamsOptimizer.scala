package utils

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import SparkMLUtils._
import ml.dmlc.xgboost4j.scala.spark.XGBoostEstimator

object HyperparamsOptimizer {


  def optimize(hyperParams: Seq[(Double, Double, Int, Double, Double, Double, Int, Int)])(implicit ss: SparkSession) = {
    import ss.implicits._

    val features = Array("eta", "numRound", "subSample", "colSampleByTree", "gamma", "min_child_weight", "max_depth")
    val observations = ss.sparkContext.parallelize(
      hyperParams
    ).toDF("rmse" +: features:_*)

    observations.show()

    def featuresAssembler = {
      new VectorAssembler()
        .setInputCols(features)
        .setOutputCol("features")
    }

    val preparedObservations = observations
      .applyTransformation(featuresAssembler)
      .select($"rmse".as("label"), $"features")

    val paramMap = List(
      "eta" -> 0.2,
      "num_round" -> 100,
      "max_depth" -> 3,
      //          "alpha" -> 0, // L2 regularization
      //          "lambda" -> 1, // L1 regularization
      "gamma" -> 0, //minimum loss reduction required to make a further partition on a leaf node of the tree. The larger, the more conservative the algorithm will be.
      "min_child_weight" -> 1.0,
      "subsample" -> 0.9,
      "colsample_bytree" -> 0.9,
      "colsample_bylevel" -> 1,
      "max_delta_step" -> 0,
      "scale_pos_weight" -> 1,
      "base_score" -> 0.5,
      "eval_metric" -> "rmse",
      "booster" -> "gbtree",
      "seed" -> 49,
      "silent" -> 1,
      "objective" -> "reg:linear").toMap

    println("Starting Xgboost ")

    val xGBoostEstimator = new XGBoostEstimator(paramMap)
    val pipeline = new Pipeline().setStages(Array(xGBoostEstimator))

    val evaluator = new RegressionEvaluator().setMetricName("rmse") // move to trait

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(new ParamGridBuilder().build())
      .setNumFolds(3)

    val cvModel = cv.fit(preparedObservations)

    val predictionsForTest = cvModel.transform(preparedObservations)

//    predictionsForTest.showAll()

    val rmse = cv.getEvaluator.evaluate(predictionsForTest)

    println("RMSE:  " + rmse)
    cvModel

  }
}

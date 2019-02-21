package com.automl.template.simple

import com.automl.PaddedLogging
import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.EvaluationMagnet
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql._
import org.apache.spark.sql.functions.col

case class NeuralNetwork(layers: Array[Int])(implicit val logPaddingSize: Int = 0) extends SimpleModelMember with PaddedLogging{
  override def name: String = "NeuralNetwork " + super.name


  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case BinaryClassificationProblem => true
    case MultiClassClassificationProblem => true
    case RegressionProblem => true
  }

  lazy val predictor = new MultilayerPerceptronClassifier()
    .setLayers(layers)
    .setBlockSize(128)
    .setSeed(1234L)
    .setMaxIter(100)

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???


  override def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType): FitnessResult = {

    val numFeatures = trainDF.select(col("features")).first().getAs[Vector](0).size

    require(layers.head == numFeatures, "Size of first layer of NN should be equal to number of features")

    val model = predictor.fit(trainDF)

    val result = model.transform(testDF)
    val predictionAndLabels = result.select("prediction", "label")

    val evaluator = new RegressionEvaluator()

    val rmse: Double = evaluator.evaluate(predictionAndLabels)

    logger.info(s"$name : RMSE = " + rmse)
    FitnessResult(Map("rmse" -> rmse), problemType,  predictionAndLabels.select("prediction"))
  }

}

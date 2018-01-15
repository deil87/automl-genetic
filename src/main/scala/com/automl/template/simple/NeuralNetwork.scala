package com.automl.template.simple

import com.automl.helper.FitnessResult
import com.automl.template.EvaluationMagnet
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

case class NeuralNetwork(layers: Array[Int]) extends SimpleModelMember {
  override def name: String = "NeuralNetwork " + super.name

  lazy val predictor = new MultilayerPerceptronClassifier()
    .setLayers(layers)
    .setBlockSize(128)
    .setSeed(1234L)
    .setMaxIter(100)

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

  override def fitnessError(trainDF: DataFrame, testDF: DataFrame): FitnessResult = {

    val model = predictor.fit(trainDF)

    val result = model.transform(testDF)
    val predictionAndLabels = result.select("prediction", "label")

    val evaluator = new RegressionEvaluator()

    val rmse: Double = evaluator.evaluate(predictionAndLabels)

    println(s"$name : RMSE = " + rmse)
    FitnessResult(rmse, predictionAndLabels.select("prediction"))
  }

}

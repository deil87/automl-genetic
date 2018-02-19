package com.stacking

import com.automl.spark.SparkSessionProvider
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{GBTRegressor, LinearRegression}
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.scalatest.{FunSuite, Matchers}


class GenericStackingSuite extends FunSuite with Matchers with SparkSessionProvider{

  import ss.implicits._
  import utils.SparkMLUtils._

  test("Should combine models for stacking") {
    val observations = ss.sparkContext.parallelize(
      Seq(
        (1, 1, 100.0),
        (2, 2, 200.0),
        (3, 3, 300.0),
        (4, 2, 200.0),
        (5, 1, 100.0),
        (6, 2, 200.0),
        (7, 1, 200.0), // <- error
        (8, 2, 300.0), // <- error
        (9, 1, 100.0),
        (10, 2, 200.0),
        (11, 1, 100.0),
        (12, 2, 200.0),
        (13, 1, 100.0),
        (14, 2, 200.0),
        (15, 1, 100.0),
        (16, 2, 200.0)
      )
    ).toDF("id", "num", "label")

    def featuresAssembler = {
      new VectorAssembler()
        .setInputCols(Array("num"))
        .setOutputCol("features")
    }

    val preparedObservations = featuresAssembler.transform(observations)


    val stacking = new GenericStacking(3)



    val Array(trainingSplit,testSplit)  = preparedObservations.randomSplit(Array(0.8, 0.2),11L)

   stacking.foldingStage(trainingSplit, testSplit)
    /*
    * Preparing DNN
    * */
    //Create the network
    val numInputs = 1
    val numOutputs = 1
    val numHiddenNodes = 10
    val numHiddenNodes2 = 20
    val seed = 12345

    //Number of iterations per minibatch
    val iterations = 1
    //Network learning rate
    val learningRate = 0.0001

    /*
    * Can't initialize before XGBoost execution
    * */
    lazy val net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(learningRate)
      //      .regularization(true)/*.l1(0.1).*/.l2(0.001).dropOut(0.5)
      .weightInit(WeightInit.XAVIER)
      //        .setGradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
      .updater(Updater.SGD)
      .momentum(0.9)
      .list
      .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
        .activation(Activation.RELU)
        .build)
      .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes2)
        .activation(Activation.TANH)
        .build)
      .layer(2, new DenseLayer.Builder().nIn(numHiddenNodes2).nOut(numHiddenNodes)
        .activation(Activation.HARDTANH)
        .build)
      .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .activation(Activation.IDENTITY)
        .nIn(numHiddenNodes).nOut(numOutputs)
        .build)
      .pretrain(false).backprop(true)
      .build)

    /*   End of preparation DNN*/

    val predictor2 = new LinearRegression()
    stacking.addModel(predictor2, trainingSplit, testSplit)

    val predictor3 = new LinearRegression().setFitIntercept(false).setRegParam(0.1)
    stacking.addModel(predictor3, trainingSplit, testSplit)
    val predictor4 = new GBTRegressor()
    stacking.addModel(predictor4, trainingSplit, testSplit)

    val parametersMap: Map[String, Any] = Map(
      "objective" -> "reg:linear",
      "booster"-> "gbtree",
      "eval_metric"-> "rmse",
      "eta"-> 0.3,
      "num_round"-> 50,
      "subsample"-> 0.9,
      "colsample_bytree"-> 0.9,
      "colsample_bylevel"-> 1.0,
      "min_child_weight"-> 1.0,
      "max_depth"-> 6,
      "max_delta_step"-> 0.0,
      "gamma"-> 0.0
    )

    stacking.addModel(net, trainingSplit, testSplit, withEarlyStoppingByScore = true)

    stacking.trainModelsPredictionsDF.showAll()
    stacking.testModelsPredictionsDF.showAll()

    val finalPredictions = stacking.performStacking(predictor4)
    finalPredictions.showAll()

  }

}


package com.automl.template.simple

import com.automl.{LogLossCustom, PaddedLogging}
import com.automl.evolution.dimension.hparameter._
import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.EvaluationMagnet
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql._
import org.apache.spark.sql.functions.col

import scala.collection.immutable
import scala.util.Random

/**
  *
  * @param hiddenLayers array of sizes for hidden layers only. Input and output layers will be computed automatically and added to array.
  */
case class NeuralNetwork(hpGroup: NeuralNetworkHPGroup = NeuralNetworkHPGroup(),
                         seed: Long = Random.nextLong())(implicit val logPaddingSize: Int = 0) extends SimpleModelMember with PaddedLogging{
  override def name: String = "NeuralNetwork " + super.name

  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case BinaryClassificationProblem => true
    case MultiClassClassificationProblem => true
    case RegressionProblem => true
  }

  var hpGroupInternal: HyperParametersGroup[_ <: MutableHParameter[Double, _]] = hpGroup

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???


  override def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType, hpFieldFromCoevolution: Option[HyperParametersField]): FitnessResult = {

    problemType match {
      case MultiClassClassificationProblem =>

        val activeHPGroup: HyperParametersGroup[_] = getRelevantHPGroupFromActiveHPField(hpFieldFromCoevolution).getOrElse(hpGroup)

        val numFeatures: Int = trainDF.select(col("features")).first().getAs[Vector](0).size
        val numLabels: Int = trainDF.select(col("indexedLabel")).distinct().collect().map(_.getDouble(0).toInt).size

        val hiddenLayers: Array[Int] = activeHPGroup.hpParameters.filter(_.isInstanceOf[HiddenLayers])
          .flatMap{ case p@HiddenLayers(_) =>
            (1 to p.currentValue.toInt).map {
              case 1 =>
                val layers: Seq[Neurons1stLayer] = activeHPGroup.hpParameters.filter(_.isInstanceOf[Neurons1stLayer]).map(_.asInstanceOf[Neurons1stLayer])
                layers(0).currentValue.toInt
              case 2 =>
                val layers: Seq[Neurons2ndLayer] = activeHPGroup.hpParameters.filter(_.isInstanceOf[Neurons2ndLayer]).map(_.asInstanceOf[Neurons2ndLayer])
                layers(0).currentValue.toInt
              case 3 =>
                val layers: Seq[Neurons3rdLayer] = activeHPGroup.hpParameters.filter(_.isInstanceOf[Neurons3rdLayer]).map(_.asInstanceOf[Neurons3rdLayer])
                layers(0).currentValue.toInt
            }
        }.toArray

        val allLayers = (numFeatures +: hiddenLayers) :+ numLabels
        val predictor = new MultilayerPerceptronClassifier()
          .setLayers(allLayers)
          .setLabelCol("indexedLabel")
          .setSeed(seed)
          .setBlockSize(128)
          .setTol(1E-6)
          .setMaxIter(20)

        val model = predictor.fit(trainDF)

        val predictionAndLabels = model.transform(testDF)

        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("indexedLabel")
          .setPredictionCol("prediction")
          .setMetricName("f1")

        val f1: Double = evaluator.evaluate(predictionAndLabels)

        val logLoss = LogLossCustom.compute(predictionAndLabels)

        val mapOfMetrics = Map("f1" -> f1, "logloss" -> logLoss)
        info(s"Finished. ${name} ${activeHPGroup.hpParameters.mkString(",")} : ${mapOfMetrics.map{ nameToValue => nameToValue._1 + " = " + nameToValue._2}.mkString("",",", "")}. Number of rows = train:${trainDF.count()} / test:${testDF.count()}")
        FitnessResult(mapOfMetrics, problemType, predictionAndLabels)
    }

  }
}

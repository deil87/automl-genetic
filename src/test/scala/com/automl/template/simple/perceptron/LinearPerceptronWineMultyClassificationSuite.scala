package com.automl.template.simple.perceptron

import com.automl.classifier.LinearPerceptronClassifier
import com.automl.spark.SparkSessionProvider
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpec}
import utils.SparkMLUtils
import com.test.TestBase


class LinearPerceptronWineMultyClassificationSuite extends WordSpec with BeforeAndAfterAll with SparkSessionProvider with TestBase{

  ss.sparkContext.setLogLevel("ERROR")
  import org.apache.spark.sql.functions._
  import ss.implicits._
  import utils.SparkMLUtils._

  println(ss.sparkContext.getConf.getAll.map(t => t._1 + ":" + t._2).mkString("\n"))

  val wineDF = SparkMLUtils.loadResourceDF("/dataset/wine.csv")
//    .showN_AndContinue(5)
    .withColumnRenamed("Nonflavanoid.phenols", "nf_flavonoid")
    .withColumnRenamed("Color.int", "color_int")

  "LinearPerceptron" should {

    "be able to separate dataset into two classes( binary case)" ignore { // TODO fix the test

      val features = Array("Mg", "Flavanoids", "nf_flavonoid", "Proanth", "color_int", "Hue", "OD", "Proline")

      val featuresColName: String = "features"

      def featuresAssembler = {
        new VectorAssembler()
          .setInputCols(features)
          .setOutputCol(featuresColName)
      }

      val scaler = new StandardScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
        .setWithStd(true)
        .setWithMean(false)

      // We are selecting only 1 and 3 classes to make it binary classification problem
      val preparedWineDF = wineDF
        .applyTransformation(featuresAssembler)
        .applyTransformation(scaler)
        .drop("features")
        .withColumnRenamed("scaledFeatures", "features")
        .withColumnRenamed("Wine", "label")
        .toDouble("label")
        .withColumnReplace("label", when($"label" === "3", 0.0).otherwise(1.0))
//        .showAllAndContinue

      val Array(trainingSplit, testSplit) = preparedWineDF.randomSplit(Array(0.8, 0.2))

      trainingSplit.cache()

      val classifier = new LinearPerceptronClassifier()
      val vectorOfParameters = classifier.trainIteratively(preparedWineDF)

      val withPredictionsDF = classifier.predict(testSplit, Seq(vectorOfParameters))

      val evaluator = new BinaryClassificationEvaluator().setRawPredictionCol("prediction")
      evaluator.evaluate(withPredictionsDF) shouldBe 1.0

    }

    "be able to separate dataset into three classes( multiclass case)" ignore {

      val features = Array("Mg", "Flavanoids", "nf_flavonoid", "Proanth", "color_int", "Hue", "OD", "Proline")

      val featuresColName: String = "features"

      def featuresAssembler = {
        new VectorAssembler()
          .setInputCols(features)
          .setOutputCol(featuresColName)
      }

      val scaler = new StandardScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
        .setWithStd(true)
        .setWithMean(false)

      // We are selecting only 1 and 3 classes to make it binary classification problem
      val preparedWineDF = wineDF
        .applyTransformation(featuresAssembler)
        .applyTransformation(scaler)
        .drop("features")
        .withColumnRenamed("scaledFeatures", "features")
        .withColumnRenamed("Wine", "label")
        .withColumnReplace("label", $"label" - 1.0 ) // TODO we need to make it automatically
        .toDouble("label")
//        .showAllAndContinue

      val Array(trainingSplit, testSplit) = preparedWineDF.randomSplit(Array(0.8, 0.2))

      trainingSplit.cache()

      val classifier = new LinearPerceptronClassifier()
      val seqOfVectorOfParameters = classifier.trainIterativelyMulticlasses(preparedWineDF)

      val withPredictionsDF = classifier.predict(testSplit, seqOfVectorOfParameters)

      val evaluator = new MulticlassClassificationEvaluator()
      val f1Measure = evaluator.evaluate(withPredictionsDF)

      println(s"F1 measure: $f1Measure")
      f1Measure shouldBe 0.9 +- 0.1

    }
  }

  override protected def afterAll(): Unit = {
//    ss.stop()
  }
}


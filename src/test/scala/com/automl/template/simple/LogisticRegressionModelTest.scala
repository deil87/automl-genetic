package com.automl.template.simple

import com.automl.spark.SparkSessionProvider
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.scalatest.{BeforeAndAfterAll, FunSuite, Matchers}
import utils.SparkMLUtils

class LogisticRegressionModelTest extends FunSuite with Matchers with BeforeAndAfterAll with SparkSessionProvider {

  ss.sparkContext.setLogLevel("ERROR")
  import org.apache.spark.sql.functions._
  import ss.implicits._
  import utils.SparkMLUtils._

  println(ss.sparkContext.getConf.getAll.map(t => t._1 + ":" + t._2).mkString("\n"))

  val wineDF = SparkMLUtils.loadResourceDF("/dataset/wine.csv") // Exploratory data analysis https://rpubs.com/alicew1800/edwine_eda3
    .showN_AndContinue(5)
    .withColumnRenamed("Nonflavanoid.phenols", "nf_flavonoid")
    .withColumnRenamed("Color.int", "color_int")

  test("be able to separate dataset into three classes( multiclass case) with LogisticRegression") {

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
      .showAllAndContinue

    val Array(trainDF, testDF) = preparedWineDF.randomSplit(Array(0.8, 0.2))

    val model = new LogisticRegression()
      .setLabelCol("label")
      .setMaxIter(500)
      .setRegParam(0.2)
      .setElasticNetParam(0.8)

    val lrModel = model.fit(trainDF)
    val prediction = lrModel.transform(testDF)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setMetricName("f1")

    val f1 = evaluator.evaluate(prediction)

    println(s"F1 measure: $f1")
    f1 shouldBe 0.9 +- 0.1

  }
}
package com.automl.dataset

import com.automl.TPopulation
import com.automl.spark.SparkSessionProvider
import com.automl.template.LeafTemplate
import com.automl.template.simple.{Bayesian, DecisionTree, LogisticRegressionModel}
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{monotonically_increasing_id, rand}
import utils.SparkMLUtils

object Datasets extends SparkSessionProvider {

  import SparkMLUtils._
  import ss.implicits._

  def getWineDataframe: DataFrame = {
    val wineDF = SparkMLUtils.loadResourceDF("/dataset/wine.csv") // Exploratory data analysis https://rpubs.com/alicew1800/edwine_eda3
      .showN_AndContinue(5)
      .withColumnRenamed("Nonflavanoid.phenols", "nf_flavonoid")
      .withColumnRenamed("Color.int", "color_int")

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
      .orderBy(rand())
      .applyTransformation(featuresAssembler)
      .applyTransformation(scaler)
      .drop("features")
      .withColumnRenamed("scaledFeatures", "features")
      .withColumnRenamed("Wine", "indexedLabel")
      .withColumnReplace("indexedLabel", $"indexedLabel" - 1.0 ) // TODO we need to make it automatically
      .withColumn("uniqueIdColumn", monotonically_increasing_id)
      .toDouble("indexedLabel")
//      .showAllAndContinue

    preparedWineDF
  }

  def getGlassDataFrame: DataFrame = {
    val glassDF = SparkMLUtils.loadResourceDF("/dataset/glass/glass.csv")

    val features = Array("RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe")

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

    val preparedGlassDF = glassDF
      .orderBy(rand())  // Shuffling
      .applyTransformation(featuresAssembler)
      .applyTransformation(scaler)
      .drop("features")
      .withColumnRenamed("scaledFeatures", "features")
      .toLong("Id")
      .withColumnRenamed("Id", "uniqueIdColumn")
      .withColumnRenamed("TypeOfGlass", "indexedLabel")
      .toDouble("indexedLabel")
      .showN_AndContinue(10)
      .cache()
    preparedGlassDF
  }
}
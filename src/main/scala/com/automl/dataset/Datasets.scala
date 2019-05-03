package com.automl.dataset

import com.automl.population.TPopulation
import com.automl.spark.SparkSessionProvider
import com.automl.template.LeafTemplate
import com.automl.template.simple.{Bayesian, DecisionTree, LogisticRegressionModel}
import org.apache.spark.ml.feature.{StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{monotonically_increasing_id, rand}
import org.apache.spark.sql.types.{IntegerType, StringType}
import utils.SparkMLUtils

object Datasets extends SparkSessionProvider {

  import SparkMLUtils._
  import ss.implicits._

  def getWineDataframe: DataFrame = {
    val wineDF = SparkMLUtils.loadResourceDF("/dataset/wine.csv") // Exploratory data analysis https://rpubs.com/alicew1800/edwine_eda3
      .showN_AndContinue(5)
      .withColumnRenamed("Nonflavanoid.phenols", "nf_flavonoid")
      .withColumnRenamed("Malic.acid", "malic_acid")
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
      .setWithMean(false) // or true?

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

  def getGlassDataFrame(shufflingSeed: Long): DataFrame = {
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

    val labelIndexer = new StringIndexer()
      .setInputCol("TypeOfGlass")
      .setOutputCol("indexedLabel")
      .setStringOrderType("alphabetAsc")

    val preparedGlassDF = glassDF
      .orderBy(rand(shufflingSeed))  // Shuffling
      .applyTransformation(featuresAssembler)
      .applyTransformation(scaler)
      .drop("features")
      .withColumnRenamed("scaledFeatures", "features")
      .toLong("Id")
      .withColumnRenamed("Id", "uniqueIdColumn")
      .applyIndexer(labelIndexer)
//      .toDouble("indexedLabel")
//      .showAllAndContinue
      .cache()
    preparedGlassDF
  }

  def getIrisDataFrame(shufflingSeed: Long): DataFrame = {

    val data = SparkMLUtils.loadResourceDF("/iris.csv")

    val labelIndexer = new StringIndexer()
      .setInputCol("fl_class")
      .setOutputCol("indexedLabel")
      .setStringOrderType("alphabetAsc")

    def basePredictorsFeaturesAssembler = new VectorAssembler()
      .setInputCols(Array("s_length", "s_width", "p_length", "p_width"))
      .setOutputCol("features")

    val preparedData = data
      .orderBy(rand(shufflingSeed))
      .applyIndexer(labelIndexer)
      .withColumn("uniqueIdColumn", monotonically_increasing_id)
        .applyTransformation(basePredictorsFeaturesAssembler)

    preparedData.showAllAndContinue

  }
}

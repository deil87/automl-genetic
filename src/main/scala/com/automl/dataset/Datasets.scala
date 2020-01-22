package com.automl.dataset

import com.automl.spark.SparkSessionProvider
import org.apache.spark.ml.feature.{StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{monotonically_increasing_id, rand}
import utils.SparkMLUtils

object Datasets extends SparkSessionProvider {

  import SparkMLUtils._
  import ss.implicits._

  import org.apache.spark.sql.functions.udf

  /**
    *
    * @param shufflingSeed to make sure we will have same data in the splits when we do `randomSplit(Array(0.8, 0.2), seed)` afterwards
    * @return
    */
  def getWineDataframe(shufflingSeed: Long): DataFrame = {
    val wineDF = SparkMLUtils.loadResourceDF("/dataset/wine.csv") // Exploratory data analysis https://rpubs.com/alicew1800/edwine_eda3
//      .showN_AndContinue(5)
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
      .orderBy(rand(shufflingSeed))
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

      preparedData
//    preparedData.showAllAndContinue

  }

  def getBalanceDataFrame(shufflingSeed: Long): DataFrame = {
    val data = SparkMLUtils.loadResourceDF("/dataset/balance_scale/balance-scale.csv")

    val labelIndexer = new StringIndexer()
      .setInputCol("className")
      .setOutputCol("indexedLabel")
      .setStringOrderType("alphabetAsc")

    def basePredictorsFeaturesAssembler = new VectorAssembler()
      .setInputCols(Array("leftWeight","leftDistance","rightWeight","rightDistance"))
      .setOutputCol("features")

    val preparedData = data
      .orderBy(rand(shufflingSeed))
      .applyIndexer(labelIndexer)
      .withColumn("uniqueIdColumn", monotonically_increasing_id)
      .applyTransformation(basePredictorsFeaturesAssembler)

    preparedData
//    preparedData.showAllAndContinue

  }

  def getCarDataFrame(shufflingSeed: Long): DataFrame = {
    val carDF = SparkMLUtils.loadResourceDF("/dataset/car/car.csv")

    val features = Array(/*"buying", "maint", "doors", */"personsEncoded", "doorsEncoded", "buyingEncoded", "maintEncoded", "lugBootEncoded"  /*"lug_boot"*/)

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
      .setInputCol("safety")
      .setOutputCol("indexedLabel")
      .setStringOrderType("alphabetAsc")

    val ordinalPersonsMapping = Map("2" -> 1, "4" -> 2, "more" -> 3 )
    val ordinalDoorsMapping = Map("2" -> 1, "3" -> 2, "4" -> 3, "5more" -> 4)
    val ordinalBuyingOrMaintMapping = Map("low" -> 1, "med" -> 2, "high" -> 3, "vhigh" -> 4)
    val ordinalLuggageBootMapping = Map("small" -> 1, "med" -> 2, "big" -> 3)

    val preparedCarDF = carDF
      .orderBy(rand(shufflingSeed))  // Shuffling
      .withColumn("buyingEncoded", udf { buyingAsStr => ordinalBuyingOrMaintMapping.get(buyingAsStr)}.apply($"buying"))
      .withColumn("maintEncoded", udf { maintAsStr => ordinalBuyingOrMaintMapping.get(maintAsStr)}.apply($"maint"))
      .withColumn("personsEncoded", udf { personsAsStr => ordinalPersonsMapping.get(personsAsStr)}.apply($"persons"))
      .withColumn("doorsEncoded", udf { doorsAsStr => ordinalDoorsMapping.get(doorsAsStr)}.apply($"doors"))
      .withColumn("lugBootEncoded", udf { lugBootAsStr => ordinalLuggageBootMapping.get(lugBootAsStr)}.apply($"lug_boot"))

      .applyTransformation(featuresAssembler)
      //      .applyTransformation(scaler)
//      .withColumnRenamed("scaledFeatures", "features")
      .withColumn("uniqueIdColumn", monotonically_increasing_id)
      .applyIndexer(labelIndexer)
//      .showAllAndContinue
      .cache()
    preparedCarDF
  }


  // REGRESSION // TODO test this DF
  def getAirlinesDataFrameRegression(shufflingSeed: Long): DataFrame = {
    import utils.SparkMLUtils._

    lazy val airlineDF = SparkMLUtils.loadParquet("src/test/resources/airline_allcolumns_sampled_100k_parquet")
      .select("DayOfWeek", "Distance", "DepTime", "CRSDepTime", "DepDelay")
    //TODO FlightNum+year_date_day for unique identifier of test examples

    val features = Array("Distance", "DayOfWeek")
    val oheFeatures = Array.empty

    val combinedFeatures = features

    val featuresColName: String = "features"

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)

    def featuresAssembler = {
      new VectorAssembler()
        .setInputCols(combinedFeatures)
        .setOutputCol(featuresColName)
    }
    import org.apache.spark.sql.functions.monotonically_increasing_id

    lazy val prepairedAirlineDF = airlineDF
      .limit(3000)
      .applyTransformation(featuresAssembler)
      .applyTransformation(scaler)
      .withColumnReplace("features", "scaledFeatures")

      .withColumnRenamed("DepDelay", "label")
      .toDouble("label")
      .filterOutNull("label")
      .withColumn("uniqueIdColumn", monotonically_increasing_id)
      //    .showN_AndContinue(100)
      .cache()
    prepairedAirlineDF
  }
}

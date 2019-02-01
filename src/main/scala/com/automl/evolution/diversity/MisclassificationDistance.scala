package com.automl.evolution.diversity

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.when

class MisclassificationDistance extends PhenotypicDistance {

  object Const {
    val cosineDistanceBasedOnErrors = "cosineDistanceBasedOnErrors"
  }

  import utils.SparkMLUtils._

  def getDistance(df1: DataFrame, df2: DataFrame): MisclassificationDistanceResult = {
    import df1.sparkSession.implicits._
    val df1Prepared = df1.select("uniqueIdColumn", "indexedLabel", "prediction").withColumnRenamed("prediction", "predictionFrom")
    val df2Prepared = df2.select("uniqueIdColumn", "prediction").withColumnRenamed("prediction", "predictionTo")

    val joineded = df1Prepared.join(df2Prepared, "uniqueIdColumn").cache()

    import org.apache.spark.sql.functions.lit
    val withMisclassificationErrors = joineded
      .withColumn("errorsFrom",  when($"indexedLabel" =!= $"predictionFrom", lit(1.0)).otherwise(lit(0.0)) )
      .withColumn("errorsFromIdx",  when($"indexedLabel" =!= $"predictionFrom", $"uniqueIdColumn").otherwise(lit(0.0)) )
      .withColumn("matchesFromIdx",  when($"indexedLabel" === $"predictionFrom", $"uniqueIdColumn").otherwise(lit(0.0)) )
      .withColumn("errorsTo",  when($"indexedLabel" =!= $"predictionTo", lit(1.0)).otherwise(lit(0.0)) )
      .withColumn("errorsToIdx",  when($"indexedLabel" =!= $"predictionTo", $"uniqueIdColumn").otherwise(lit(0.0)) )
      .withColumn("matchesToIdx",  when($"indexedLabel" === $"predictionTo", $"uniqueIdColumn").otherwise(lit(0.0)) )
      .withColumn("similarMistakesDist",  similarMistakesDistance($"indexedLabel", $"predictionFrom", $"predictionTo") )
      .withColumn("nonSimilarMistakesDist",  nonSimilarMistakesDistance($"indexedLabel", $"predictionFrom", $"predictionTo") )
      .withColumn("componentsForCosine",  cosineDistanceOnErrors($"indexedLabel", $"predictionFrom", $"predictionTo") )
      .cache()

    val cosineDistanceBasedOnErrors = {
      val components = withMisclassificationErrors.select("componentsForCosine")
        .map(_.getAs[DenseVector](0).values)
        .rdd
        .reduce((arr1, arr2) =>  Array(arr1(0) + arr2(0), arr1(1) + arr2(1), arr1(2) + arr2(2) ))

      val dotProduct = components(0)
      val norm1 = components(1)
      val norm2 = components(2)
      val cosineSimilarity = dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2))
      val cosineDistance = 1.0 - cosineSimilarity
      if(cosineDistance < 1e-5) 0 else cosineDistance
    }

    // Calculating errorsOnSamePositionsCount
    val positionsOfMistakesOnDF1 = withMisclassificationErrors.select("errorsFromIdx").filter($"errorsFromIdx" =!= 0).map(_.getDouble(0)).collect()
    val positionsOfMistakesOnDF2 = withMisclassificationErrors.select("errorsToIdx").filter($"errorsToIdx" =!= 0).map(_.getDouble(0)).collect()
    val errorsOnSamePositionsCount = positionsOfMistakesOnDF1.intersect(positionsOfMistakesOnDF2).length

    val positionsOfMatchesOnDF1 = withMisclassificationErrors.select("matchesFromIdx").filter($"matchesFromIdx" =!= 0).map(_.getDouble(0)).collect()
    val positionsOfMatchesOnDF2 = withMisclassificationErrors.select("matchesToIdx").filter($"matchesToIdx" =!= 0).map(_.getDouble(0)).collect()
    val matchesOnSamePositionsCount = positionsOfMatchesOnDF1.intersect(positionsOfMatchesOnDF2).length


    val vectorForFinalCosineSimilarity = new VectorAssembler()
      .setInputCols(Array(Const.cosineDistanceBasedOnErrors, "similarMistakesDist", "nonSimilarMistakesDist"))
      .setOutputCol("finalComponents")

    val withCosineDistance = withMisclassificationErrors
      .drop("componentsForCosine")
      .withColumn(Const.cosineDistanceBasedOnErrors,  lit(cosineDistanceBasedOnErrors ))
      .applyTransformation(vectorForFinalCosineSimilarity)
      .showN_AndContinue(100)


    MisclassificationDistanceResult(cosineDistanceBasedOnErrors, errorsOnSamePositionsCount, matchesOnSamePositionsCount, withCosineDistance)
  }


  // Think about case when one is right and one is wrong
  def nonSimilarMistakesDistance = {
    import org.apache.spark.sql.functions.udf
    udf {( trueLabel: Double,  predictionFrom: Double,  predictionTo: Double) =>
      if (trueLabel != predictionFrom && trueLabel != predictionTo && predictionFrom != predictionTo) {
        new DenseVector( Array( trueLabel - predictionFrom, trueLabel - predictionTo))
      } else {
        new DenseVector(Array(0.0, 0.0))
      }
    }
  }

  // We need to conpute on whole data/rows as we want out distances between different predictors to be comparable and unambiguous.
  def similarMistakesDistance = {
    import org.apache.spark.sql.functions.udf
    udf {( trueLabel: Double,  predictionFrom: Double,  predictionTo: Double) =>
      if (trueLabel != predictionFrom && predictionFrom == predictionTo) {
        trueLabel - predictionFrom
      } else {
        0.0
      }
    }
  }

  def cosineDistanceOnErrors = {
    import org.apache.spark.sql.functions.udf
    udf {( trueLabel: Double,  predictionFrom: Double,  predictionTo: Double) =>
      val errorFrom = if(trueLabel != predictionFrom) 1.0 else 0.0
      val errorTo = if(trueLabel != predictionTo) 1.0 else 0.0
      val dotProductComponent = errorFrom * errorTo
      val norm1Component = errorFrom
      val norm2Component = errorTo
      new DenseVector(Array( dotProductComponent, norm1Component, norm2Component))
    }
  }

  // Most important discrimination is based on cosineDistanceOnErrors
  /*def hyperCosineDistance = {
    import org.apache.spark.sql.functions.udf
    udf { components: DenseVector =>
      val Array(cosineDistanceOnErrors,  similarMistakesDist,  nonSimilarMistakesDist_first,  nonSimilarMistakesDist_second) = components.values
      val dotProductComponent = errorFrom * errorTo
      val norm1Component = errorFrom
      val norm2Component = errorTo
      new DenseVector(Array( dotProductComponent, norm1Component, norm2Component))
    }
  }*/

  /*
    This version assumes we used a collect() method and it is not scalable solution. Consider transforming into IndexedRowMatrix and udf function.
   */
  def cosineSimilarity(vectorA: Array[Double], vectorB: Array[Double]): Double = {
    var dotProduct = 0.0
    var normA = 0.0
    var normB = 0.0
    var i = 0
    while ( {
      i < vectorA.length
    }) {
      dotProduct += vectorA(i) * vectorB(i)
      normA += Math.pow(vectorA(i), 2)
      normB += Math.pow(vectorB(i), 2)

      {
        i += 1; i - 1
      }
    }
    dotProduct / (Math.sqrt(normA) * Math.sqrt(normB))
  }
}

case class MisclassificationDistanceResult(distance: Double, errorsOnSamePositionsCount: Long, matchesOnSamePositionsCount: Long, rawData: DataFrame)

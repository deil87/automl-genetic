package com.automl

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.{Vector => MLVector}

object LogLossCustom {

  /**
    * Compute based on 'prediction' and 'label' columns
    * Should contain prediction Vec column with predictions on per class basis
   */
  def compute(observations: DataFrame, probabilitiesColumnName: String = "probability"):Double = {
    import observations.sparkSession.implicits._
    import org.apache.spark.sql.functions._

    val sum = observations.withColumn("logloss", logLossUDF($"$probabilitiesColumnName", $"indexedLabel"))
//    val sum = observations.withColumn("logloss", log($"prediction" + 1) - $"label")

//    sum.agg(sum("logloss")).first().getDouble(0)
    sum.select(col("logloss")).rdd.map(_(0).asInstanceOf[Double]).reduce(_+_) / sum.count()
  }

  def logLossUDF = {
    import org.apache.spark.sql.functions.udf
    val epsilon: Double = 1e-15
    def minMax(confidence: Double) = Math.max(Math.min(confidence, 1.0 - epsilon), epsilon)

    udf { (preds: MLVector, labelIndex: Int) =>
      val smoothedConfidence = minMax( preds.apply(labelIndex))
      -Math.log(smoothedConfidence)
    }
  }

}

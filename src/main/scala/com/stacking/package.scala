package com

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.factory.Nd4j

package object stacking {

  implicit def multiLayerNetworkHelper(net: MultiLayerNetwork) = new {

    def transform(df: DataFrame, params: DNNIteratorParams, predictionColName:String = "prediction"): DataFrame = {

      val validationFoldConverted = df.select("id","features").collect()
        .map{row =>
          (row.getInt(0), Nd4j.create( row.getAs[Vector](1).toArray , Array[Int](1, params.numInputs)))
        }

      val predictions = validationFoldConverted.map{ case (id, fts) =>
        val out = net.output(fts, false)
        (id, out)
      }

      val sparkSession: SparkSession = df.sparkSession
      import sparkSession.implicits._
      sparkSession.sparkContext.parallelize(
        predictions.map{ case (id, pred) => (id, pred.data.asDouble()(0))}
      ).toDF("id", predictionColName)
    }
  }
}

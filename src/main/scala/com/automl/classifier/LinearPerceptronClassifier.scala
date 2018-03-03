package com.automl.classifier

import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Row}

class LinearPerceptronClassifier {

  def extractFeaturesMatrix(featuresDF: DataFrame): IndexedRowMatrix = {

    val featuresMatrix: IndexedRowMatrix = new IndexedRowMatrix(
      featuresDF.rdd.map {
        case Row(a)  =>
          val vector = a.asInstanceOf[DenseVector]  // TODO whether it is true regardless construction of df approach?
          org.apache.spark.mllib.linalg.Vectors.fromML(vector)
      }.zipWithIndex.map { case (v, i) => IndexedRow(i, v) })

    featuresMatrix
  }

  def train(df: DataFrame) = {

    val featuresDF = df.select("features")
    val numFeatures = featuresDF.first().getAs[Vector](0).size

    val featuresMatrix: IndexedRowMatrix = extractFeaturesMatrix(df)


    val vectorOfParameters =  Matrices.dense(numFeatures, 1, Array(4.0, 1.0, 5.0))
//    val activation = featuresMatrix.multiply(vectorOfParameters).rows.collect().sum //+ bias


  }

  def transferFunction(activation: Double): Double = {
    if(activation >= 0.0) 1.0 else 0.0
  }


  /*def multiply(B: Matrix): RowMatrix = {
    val n = numCols().toInt
    val k = B.numCols
    require(n == B.numRows, s"Dimension mismatch: $n vs ${B.numRows}")

    require(B.isInstanceOf[DenseMatrix],
      s"Only support dense matrix at this time but found ${B.getClass.getName}.")

    ...
  }*/
}

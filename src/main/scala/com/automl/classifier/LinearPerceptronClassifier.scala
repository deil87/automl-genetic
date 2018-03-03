package com.automl.classifier

import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import utils.LabeledVector
import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.mllib.linalg.{Vector => VectorMLLib}

import scala.util.control.Breaks._
import scala.util.Random

class LinearPerceptronClassifier {

  def extractFeaturesMatrix(featuresDF: DataFrame, withBias: Boolean = false): IndexedRowMatrix = {

    val featuresMatrix: IndexedRowMatrix = new IndexedRowMatrix(
      featuresDF.rdd.map {
        case Row(a)  =>
          val vector = a.asInstanceOf[DenseVector]  // TODO whether it is true regardless construction of df approach?
          org.apache.spark.mllib.linalg.Vectors.fromML(vector)
      }.zipWithIndex.map { case (v, i) =>
        if(withBias)
          IndexedRow(i, Vectors.dense(1, v.toArray: _*)) // TODO maybe not the most efficient concatenation. Look for analog of DenseVector.vertcat(a,b)
        else
          IndexedRow(i, v)
      }
    )

    featuresMatrix
  }

  /**
    *
    * @param df
    * @return vector of parameters
    */
  def trainIteratively(df: DataFrame): VectorMLLib = {
    import df.sparkSession.implicits._

    val input = df.as[LabeledVector]
    input.cache()

    val numFeatures = input.head().features.size + 1 // one for bias

    var vectorOfParameters = Vectors.dense(Array.fill(numFeatures)(Random.nextDouble()))


    val elementwiseAddition: (Array[Double], Array[Double]) => Array[Double] = { (x:Array[Double], y: Array[Double]) =>
      require(x.length == y.length)
      x.zip(y).map{ case(xn, yn) => xn + yn}
    }

    val elementwiseSubtraction: (Array[Double], Array[Double]) => Array[Double]  = { (x:Array[Double], y: Array[Double]) =>
      require(x.length == y.length)
      x.zip(y).map{ case(xn, yn) => xn - yn}
    }

    /**
      *
      * @ classOfExample 1 - for positive real label, 0 - for negative label
      * @return
      */
    val transferFunction: (Double, Int) => ((VectorMLLib, VectorMLLib) => VectorMLLib, Boolean) = { (activation: Double, classOfExample: Int) =>
      def add: (VectorMLLib, VectorMLLib) => VectorMLLib = (x: VectorMLLib, y: VectorMLLib) => {
        val res = Vectors.dense(elementwiseAddition(x.toArray,y.toArray))
        res
      }
      def subtr: (VectorMLLib, VectorMLLib) => VectorMLLib = (x: VectorMLLib, y: VectorMLLib) => Vectors.dense(elementwiseSubtraction(x.toArray,y.toArray))
      def nothing: (VectorMLLib, VectorMLLib) => VectorMLLib = (x: VectorMLLib, y: VectorMLLib) => x

      var misclassified = false
      val action =
        if (classOfExample == 1)
          if (activation <= 0.0) {
            misclassified = true
            add
          }
          else nothing // add features vector to parameters
        else if (activation >= 0.0) {
          misclassified = true
          subtr
        }
        else nothing // subtract features vector to parameters
      (action, misclassified)
    }

    val calculateAction: LabeledVector => ((VectorMLLib, VectorMLLib) => VectorMLLib, Boolean) = { row: LabeledVector =>

      // val activation = featuresAsBreeze.dot(vectorOfParametersAsBreeze) //TODO breeze version of .dot is not serializable and can't be used with Spark
      // Uses netlib-native_system-osx-x86_64.jnilib
      val activation = Matrices.dense(numFeatures, 1, Array(1, row.features.toArray:_* )).transpose.multiply(vectorOfParameters) // or we can reshape?

      val activationValue = activation.values(0)
      println("AV:" + activationValue)
      transferFunction(activationValue, row.label.toInt)
    }

    def terminationCriteria: Boolean = {
      val res = input.map { row =>
        calculateAction(row)._2 // true when misclassified.
      }
      val ret = !res.collect().forall(_ == false)
      ret
    }


    while(terminationCriteria) {
      input.foreach { row => // TODO in original algorithm rows are taken randomly

        val featuresWithBias: Array[Double] = Array(1.0, row.features.toArray:_*) //??? cast
        val vectorOfFeaturesWithBias: VectorMLLib = Vectors.dense(featuresWithBias)

        val (learningAction, _) =  calculateAction(row)

        //TODO vectorOfParameters is not updated because it is on agents
        // 1) use map and then apply all updates at once after map is finished
        // 2) use spark streaming
        vectorOfParameters = learningAction(vectorOfParameters, vectorOfFeaturesWithBias)
        println(vectorOfParameters.toArray.mkString(","))
      }
    }
    vectorOfParameters
  }



}

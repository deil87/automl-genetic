package com.automl.classifier

import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.sql.functions.{col, monotonically_increasing_id}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import utils.{LabeledVector, UnlabeledVector}
import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.mllib.linalg.{Vector => VectorMLLib}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder

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

    val asBreeze: org.apache.spark.ml.linalg.Vector => BDV[Double] = vector => new BDV[Double](vector.toArray)
    val asBreezeFromArray: Array[Double] => BDV[Double] = vector => new BDV[Double](vector)

    val vectorOfParameters = new breeze.linalg.DenseVector[Double](Array.fill(numFeatures)(Random.nextDouble()))


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
    val transferFunction: (Double, Int) => ((BDV[Double], BDV[Double]) => BDV[Double], Boolean) = { (activation: Double, classOfExample: Int) =>
      def add: (BDV[Double], BDV[Double]) => BDV[Double] = (x, y) => {
        x += y
      }
      def subtr: (BDV[Double], BDV[Double]) => BDV[Double] = (x, y) => x -= y
      def nothing: (BDV[Double], BDV[Double]) => BDV[Double] = (x, y) => x

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

    val calculateAction: LabeledVector => ((BDV[Double], BDV[Double]) => BDV[Double], Boolean)  = { row: LabeledVector =>

      val featuresAsBreeze = asBreezeFromArray(Array(1, row.features.toArray:_* ))
      val activation = featuresAsBreeze.dot(vectorOfParameters)

      println("Activation value:" + activation)
      transferFunction(activation, row.label.toInt)
    }

    def terminationCriteria: Boolean = {
      val res = input.map { row =>
        calculateAction(row)._2 // true when misclassified.
      }
      val currentStateOfClassfication = res.collect()
      println("Number of misclassifications: " + currentStateOfClassfication.count(_ == true))
      !currentStateOfClassfication.forall(_ == false)
    }

//    implicit val myBDVEncoder = org.apache.spark.sql.Encoders.kryo[((BDV[Double], BDV[Double]) => BDV[Double], Boolean)]
    implicit val myBDVEncoder = ExpressionEncoder()


    while(terminationCriteria) { // TODO checking termination criteria once per dataset size is not efficient approach
      val learningActions = input.sample(withReplacement = false, 1).map { row =>

        val featuresWithBias: Array[Double] = Array(1.0, row.features.toArray:_*)
        val trainingVectorOfFeaturesWithBias = asBreezeFromArray(featuresWithBias)

        val (learningAction, _) =  calculateAction(row)

        //TODO vectorOfParameters is not updated because it is on agents
        // 1) use map and then apply all updates at once after map is finished
        // 2) use spark streaming with state ??
        (learningAction, trainingVectorOfFeaturesWithBias)
      }
      learningActions.collect().foreach { case (action, trainingVectorOfFeaturesWithBias) =>
        action(vectorOfParameters, trainingVectorOfFeaturesWithBias)}
    }
    //Converting back to MLLib vector
    Vectors.dense(vectorOfParameters.data)
  }

  /**
    *
    * @param unlabeledDF
    * @return df with one extra column `prediction`
    */
  def predict(unlabeledDF: DataFrame, vectorOfParameters: VectorMLLib): DataFrame = {

    import unlabeledDF.sparkSession.implicits._

    val unlabeledDfWithId = unlabeledDF
      .withColumn("id", monotonically_increasing_id)

    val asBreeze: org.apache.spark.ml.linalg.Vector => BDV[Double] = vector => new breeze.linalg.DenseVector[Double](vector.toArray)

    val unlabeledInput = unlabeledDfWithId.as[UnlabeledVector]
    unlabeledInput.cache()

    val numFeatures = unlabeledInput.head().features.size + 1

    val calculateAction: UnlabeledVector => Double = { row: UnlabeledVector =>

      val featuresAsBreeze = asBreeze(row.features)
      val activationBreeze = featuresAsBreeze.dot(featuresAsBreeze)
      val activation = Matrices.dense(numFeatures, 1, Array(1, row.features.toArray:_* )).transpose.multiply(vectorOfParameters)

      val activationValue = activation.values(0)
      println("Activation value for test example:" + activationValue)
      if(activationValue > 0)
        1.0
      else
        0.0
    }

    val featuresWithPredictions: Dataset[FeaturesVectorWithPredictions] = unlabeledInput.map { row =>
      val prediction: Double = calculateAction(row)
      FeaturesVectorWithPredictions(row.id, row.features, prediction)
    }
    featuresWithPredictions.toDF().join( unlabeledDfWithId.drop("features"), Seq("id"), joinType="left_outer" )
  }

}

final case class FeaturesVectorWithPredictions(id: Double, features: org.apache.spark.ml.linalg.Vector, prediction: Double)

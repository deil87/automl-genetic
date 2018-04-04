package com.automl.classifier

import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.sql.functions.{col, monotonically_increasing_id}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import utils.{LabeledVector, UnlabeledVector}
import org.apache.spark.mllib.linalg.{Vector => VectorMLLib}

import scala.util.Random

import utils.SparkMLUtils._

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

  def getNumberOfClasses(df: DataFrame): Int = {
    df.select("label").distinct().count().toInt
  }

  def toOneHotRepresentationForTargetVariable(df: DataFrame): DataFrame = {
    val encoder = new OneHotEncoder()
      .setDropLast(false)
      .setInputCol("label")
      .setOutputCol("labelOH")

    val encoded = encoder.transform(df)

    import org.apache.spark.sql.functions._
    import df.sqlContext.sparkSession.implicits._
    val  extractFun:  (Vector, Int) => Double = (x, i) => x.toArray(i)
    val extractDoubleUDF = udf(extractFun)

    encoded.show()

    val withSeparateLabelColumns = (0 until getNumberOfClasses(df)).foldLeft(encoded) {
      case (dataFrame, classIndex) =>
        dataFrame.withColumn(s"label_$classIndex", extractDoubleUDF($"labelOH", lit(classIndex)))
    }.drop("labelOH")
    withSeparateLabelColumns.show() // TODO remove
    withSeparateLabelColumns
  }

  def trainIterativelyMultyclasses(df: DataFrame): Seq[VectorMLLib] = {
    val numberOfClasses = getNumberOfClasses(df)
    val arrayOfVectorOfParameters: Seq[VectorMLLib] = if(numberOfClasses > 2) {
      val withSeparatedLabelsPerClass = toOneHotRepresentationForTargetVariable(df)
      withSeparatedLabelsPerClass.showAll()  //TODO remove
      withSeparatedLabelsPerClass.cache()
      ( 0 until numberOfClasses)
        .map { focusClass =>
          println(s"Calculating parameters for class $focusClass vs others")
          val vectorOfParameters = trainIteratively(withSeparatedLabelsPerClass.withColumnReplace("label", s"label_$focusClass"))
          println(s"Class $focusClass was successfully linearly separated from others")
          vectorOfParameters
        }
    }
    else {
      Seq(trainIteratively(df))
    }
    arrayOfVectorOfParameters
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


    val elementWiseAddition: (Array[Double], Array[Double]) => Array[Double] = { (x:Array[Double], y: Array[Double]) =>
      require(x.length == y.length)
      x.zip(y).map{ case(xn, yn) => xn + yn}
    }

    val elementWiseSubtraction: (Array[Double], Array[Double]) => Array[Double]  = { (x:Array[Double], y: Array[Double]) =>
      require(x.length == y.length)
      x.zip(y).map{ case(xn, yn) => xn - yn}
    }

    /**
      *
      * @ classOfExample 1 - for positive real label, 0 - for negative label
      * @return
      */
    val transferFunction: (Double, Int) => (Int, Boolean) = { (activation: Double, classOfExample: Int) =>

      var misclassified = false
      val action =
        if (classOfExample == 1)
          if (activation <= 0.0) {
            misclassified = true
            1
          }
          else 0 // add features vector to parameters
        else if (activation >= 0.0) {
          misclassified = true
          -1
        }
        else 0 // subtract features vector to parameters
      (action, misclassified)
    }

    val calculateAction: LabeledVector => (Int, Boolean) = { row: LabeledVector =>

      // val activation = featuresAsBreeze.dot(vectorOfParametersAsBreeze)
      // Uses netlib-native_system-osx-x86_64.jnilib
      val activation = Matrices.dense(numFeatures, 1, Array(1, row.features.toArray:_* )).transpose.multiply(vectorOfParameters) // or we can reshape?

      val activationValue = activation.values(0)
      println("Activation value:" + activationValue)
      transferFunction(activationValue, row.label.toInt)
    }

    var minNumberOfMisclassifications = input.count()
    var numberOfUnsuccessfulLearningIterations = 0

    def terminationCriteria: Boolean = {
      val classifiedInput = input.map { row =>
        calculateAction(row)._2 // ._2 == true when misclassified.
      }
      val currentStateOfClassification = classifiedInput.collect()
      val numberOfMissclassifications = currentStateOfClassification.count(_ == true)
      if(numberOfMissclassifications < minNumberOfMisclassifications) {
        numberOfUnsuccessfulLearningIterations = 0
        minNumberOfMisclassifications = numberOfMissclassifications
      }
      else numberOfUnsuccessfulLearningIterations += 1
      println("Number of misclassifications: " + numberOfMissclassifications)
      !currentStateOfClassification.forall(_ == false)
    }


    while(terminationCriteria && numberOfUnsuccessfulLearningIterations < 50) { // TODO checking termination criteria once per dataset size is not efficient approach
      val learningActions = input.sample(withReplacement = false, 1).map { row =>

        val featuresWithBias: Array[Double] = Array(1.0, row.features.toArray:_*)
        val vectorOfFeaturesWithBias: VectorMLLib = Vectors.dense(featuresWithBias)

        val (learningAction, _) =  calculateAction(row)

        //TODO vectorOfParameters is not updated because it is on agents
        // 1) use map and then apply all updates at once after map is finished
        // 2) use spark streaming with state ??
        (learningAction, vectorOfFeaturesWithBias)
      }
      learningActions.collect().foreach { case (action, vectorOfFeaturesWithBias) =>
        if(action == 1)
          vectorOfParameters = Vectors.dense(elementWiseAddition(vectorOfParameters.toArray, vectorOfFeaturesWithBias.toArray))
        if(action == -1)
          vectorOfParameters = Vectors.dense(elementWiseSubtraction(vectorOfParameters.toArray, vectorOfFeaturesWithBias.toArray))
      }
    }
    vectorOfParameters
  }

  /**
    *
    * @param unlabeledDF
    * @return df with one extra column `prediction`
    */
  def predict(unlabeledDF: DataFrame, seqOfTrainedParameters: Seq[VectorMLLib]): DataFrame = {

    import unlabeledDF.sparkSession.implicits._

    val unlabeledDfWithId = unlabeledDF
      .withColumn("id", monotonically_increasing_id)

    val unlabeledInput = unlabeledDfWithId.as[UnlabeledVector]
    unlabeledInput.cache()

    val numFeatures = unlabeledInput.head().features.size + 1

    val calculateAction: (UnlabeledVector, VectorMLLib) => Double = {
      (row: UnlabeledVector, vectorOfParameters:VectorMLLib) =>

        // val activation = featuresAsBreeze.dot(vectorOfParametersAsBreeze) //TODO breeze version of .dot is not serializable and can't be used with Spark
        val activation = Matrices.dense(numFeatures, 1, Array(1, row.features.toArray:_* )).transpose.multiply(vectorOfParameters)

        val activationValue = activation.values(0)
        println("Activation value for test example:" + activationValue)
        if(activationValue > 0)
          1.0
        else
          0.0
    }

    def featuresWithPredictionsBy(parameters:VectorMLLib) : Dataset[FeaturesVectorWithPredictions] = unlabeledInput.map { row =>
      val prediction: Double = calculateAction(row, parameters)
      FeaturesVectorWithPredictions(row.id, row.features, prediction)
    }

    /*  Splitting into two cases 1) Binary 2) MultiClass  */
    val featuresWithPredictions = if(seqOfTrainedParameters.length > 1) {
      val predictionsFromEachPerceptron = seqOfTrainedParameters.map{ vectorOfParameters =>
        featuresWithPredictionsBy(vectorOfParameters)
      }
      val joinedPredictionsDF = predictionsFromEachPerceptron.zipWithIndex
        .map{ case (pred, index) =>
          pred.toDF().drop("features").withColumnRenamed("prediction", s"prediction_$index")
        }
        .reduceLeft((acc: DataFrame, dfWithParticularPerceptronPrediction: DataFrame) =>
          acc.join(dfWithParticularPerceptronPrediction, "id")
        )

      import org.apache.spark.sql.functions._
      import unlabeledDF.sqlContext.sparkSession.implicits._
      val  maxFun:  (Double, Double) => Double = (c1, c2) => Math.max(c1, c2)
      val selectMaxColumn = udf(maxFun)

      /* Preparing to find maximum column between all perceptron's predictions/actions_values */
      val withfirstPredictionAsMax = joinedPredictionsDF.withColumn("maxPrediction", $"prediction_0")

      val numberOfClasses = seqOfTrainedParameters.length
      (1 until numberOfClasses)
        .foldLeft(withfirstPredictionAsMax) { case (acc, focusClass) =>
          acc.withColumnReplace("maxPrediction", selectMaxColumn($"maxPrediction", $"prediction_$focusClass"))
        }.withColumnRenamed("maxPrediction", "prediction")
    } else {
      featuresWithPredictionsBy(seqOfTrainedParameters.head)
    }

    featuresWithPredictions.toDF().join( unlabeledDfWithId.drop("features"), Seq("id"), joinType="left_outer" ) //TODO do we need this here?
  }

}

final case class FeaturesVectorWithPredictions(id: Double, features: org.apache.spark.ml.linalg.Vector, prediction: Double)

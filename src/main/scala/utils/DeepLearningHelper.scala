package utils

import java.util.{Collections, Random}

import com.automl.classifier.ensemble.stacking.DNNIteratorParams
import org.apache.spark.sql._
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.earlystopping.EarlyStoppingResult
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

object DeepLearningHelper {

  def trainingIteratorFromDF(df: DataFrame, dnnParams: DNNIteratorParams) = {

    val seed = 4567
    val rnd = new Random(seed)
    import dnnParams._
    import org.apache.spark.ml.linalg.{DenseMatrix, Vector => MLVector}

    val featuresToLabels = df.select("features", "label").collect()
    val features = featuresToLabels.flatMap(_.getAs[MLVector](0).toArray)

    val mtx = new DenseMatrix( numInputs, featuresToLabels.length, features).transpose
    val transposedFeatures = mtx.colIter.toArray.map(vec => Nd4j.create(vec.toArray, Array[Int]( featuresToLabels.length, 1)))

    val labels: Array[Double] = featuresToLabels.map(_.getDouble(1))

    val inputNDArray = Nd4j.hstack(transposedFeatures:_*)
    val outPut = Nd4j.create(labels, Array[Int](labels.length, 1))
    val dataSet = new DataSet(inputNDArray, outPut)
    val listDs = dataSet.asList
    Collections.shuffle(listDs, rnd)
    new ListDataSetIterator(listDs, batchSize)
  }


  def printLearningProcessInfo(trainingResult: EarlyStoppingResult[MultiLayerNetwork]): Unit = {
    println("Termination reason: " + trainingResult.getTerminationReason)
    println("Termination details: " + trainingResult.getTerminationDetails)
    println("Total epochs: " + trainingResult.getTotalEpochs)
    println("Best epoch number: " + trainingResult.getBestModelEpoch)
    println("Score at best epoch: " + trainingResult.getBestModelScore)
  }

}

package com.automl.spark

import com.automl.dataset.{Datasets, Stratification, StratifiedSampling}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpec}
import utils.TestBase

import scala.util.Random

class RandomGridSearchTest extends WordSpec with BeforeAndAfterAll with SparkSessionProvider with TestBase{

  import utils.SparkMLUtils._

  "Random grid search" ignore {

    // This test is fine but we can't guarantee that after grid search of hps we will do better on testSplit as we might overfit to trainingSplit during crosss validation
    "verify that we can't guarantee better performance with RGS on training split" in {

      var avgWithRGS = 0.0
      var avgBaseline = 0.0

      val numberOfRestarts = 3
      for(i <- 0 until numberOfRestarts) {
        val seed = new Random().nextLong()

        val preparedGlassDF = Datasets.getGlassDataFrame(seed).sampleRand(80, seed)
        preparedGlassDF.cache()

        val Array(trainingSplit, testSplit) = new Stratification().split(preparedGlassDF, 0.8, seed)

        trainingSplit.cache()
        testSplit.cache()

//        trainingSplit.showAllAndContinue
//        testSplit.showAllAndContinue

        val dtr = new DecisionTreeClassifier()
          .setLabelCol("indexedLabel")

        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("indexedLabel")
          .setPredictionCol("prediction")
          .setMetricName("f1")

        val paramGrid = new ParamGridBuilder()
          .addGrid(dtr.maxDepth, Array(3, 5, 7))
          .build()

        val cv = new CrossValidator()
          .setEstimator(dtr)
          .setEvaluator(evaluator)
          .setEstimatorParamMaps(paramGrid)
          .setNumFolds(5)

        val model = cv.fit(trainingSplit)

        println(model.asInstanceOf[CrossValidatorModel].avgMetrics.mkString(","))
        println("Learned classification tree model:\n" + model.asInstanceOf[CrossValidatorModel].bestModel.asInstanceOf[DecisionTreeClassificationModel].toDebugString)

        val predictions = model.transform(testSplit)

        val f1WithRGS: Double = evaluator.setMetricName("f1").evaluate(predictions)
        avgWithRGS += f1WithRGS

        //Baseline model
        val baselineModel = dtr.fit(trainingSplit)

        val predictionsBaseline = baselineModel.transform(testSplit)

        val f1WithBaseline: Double = evaluator.setMetricName("f1").evaluate(predictionsBaseline)
        avgBaseline += f1WithBaseline

        println(s"Seed = $seed ; f1WithRGS = $f1WithRGS ; f1WithBaseline = $f1WithBaseline")
      }

      avgWithRGS /= numberOfRestarts
      avgBaseline /= numberOfRestarts
      println(s"avgWithRGS = $avgWithRGS ; avgBaseline = $avgBaseline")

      avgWithRGS > avgBaseline shouldBe true

    }
  }
}

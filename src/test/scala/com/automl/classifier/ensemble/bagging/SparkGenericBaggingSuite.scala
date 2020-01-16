package com.automl.classifier.ensemble.bagging

import com.automl.ConfigProvider
import com.automl.dataset.Datasets
import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.helper.TemplateTreeHelper
import com.automl.problemtype.ProblemType
import com.automl.spark.SparkSessionProvider
import com.automl.template._
import com.automl.template.simple._
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.bytedeco.javacpp.opencv_ml.LogisticRegression
import org.scalatest.{FunSuite, Matchers}
import utils.SparkMLUtils

import scala.util.Random


class SparkGenericBaggingSuite extends FunSuite with Matchers with SparkSessionProvider{

  import utils.SparkMLUtils._
  import org.apache.spark.sql.functions.monotonically_increasing_id

  test("Spark Bagging should calculate over complex tree algorithm") {
    val metric = "logloss"
    ConfigProvider.clearOverride.addOverride(
      s"""
         |evolution {
         |  hyperParameterDimension {
         |    enabled = false
         |  }
         |  evaluation {
         |    multiclass.metric = "$metric"
         |  }
         |}
      """)
    val models = Seq(
      LeafTemplate(LogisticRegressionModel()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(RandomForest()),
      LeafTemplate(Bayesian()),
      NodeTemplate(SparkGenericBagging(), Seq(
        LeafTemplate(LogisticRegressionModel()()),
//        LeafTemplate(GradientBoosting()), //TODO
        LeafTemplate(DecisionTree())
      ))
    )

    val ensemb = NodeTemplate(SparkGenericBagging(), models)

    val data = Datasets.getIrisDataFrame(1234)

    println(TemplateTreeHelper.renderAsString_v2(ensemb))

    val Array(trainingSplit, testSplit) = data.randomSplit(Array(0.8, 0.2))

    val result = ensemb.evaluateFitness(trainingSplit, testSplit, ProblemType.MultiClassClassificationProblem, hpFieldFromCoevolution = Some(HyperParametersField.default))
    println("Logloss: " + result.getCorrespondingMetric)
  }


  test("Spark Bagging computations should be seed-reproducible") {
    val metric = "logloss"
    ConfigProvider.clearOverride.addOverride(
      s"""
         |evolution {
         |  hyperParameterDimension {
         |    enabled = false
         |  }
         |  evaluation {
         |    multiclass.metric = "$metric"
         |  }
         |}
      """)

    val ensemb = NodeTemplate(SparkGenericBagging(), Seq(
      LeafTemplate(LogisticRegressionModel()),
      LeafTemplate(DecisionTree())
    ))

    val seedForEverything = 1234
    val data = Datasets.getIrisDataFrame(seedForEverything)

    println(TemplateTreeHelper.renderAsString_v2(ensemb))

    val Array(trainingSplit, testSplit) = data.randomSplit(Array(0.8, 0.2), seedForEverything)

    val result = ensemb.evaluateFitness(trainingSplit, testSplit, ProblemType.MultiClassClassificationProblem, hpFieldFromCoevolution = Some(HyperParametersField.default), seed = seedForEverything)

    // ^^^ Should be equal to result2 from same actions below:
    val data2 = Datasets.getIrisDataFrame(seedForEverything)

    val Array(trainingSplit2, testSplit2) = data2.randomSplit(Array(0.8, 0.2), seedForEverything)

    val result2 = ensemb.evaluateFitness(trainingSplit2, testSplit2, ProblemType.MultiClassClassificationProblem, hpFieldFromCoevolution = Some(HyperParametersField.default), seed = seedForEverything)

    result.getCorrespondingMetric shouldBe result2.getCorrespondingMetric
  }

  //TODO fix the test
  ignore("Spark Bagging should sample training dataset for submembers in appropriate way )") {
    import ss.implicits._
    val threeDistinctLevelsDF = ss.sparkContext.parallelize(
      Seq(
        (1, "level1"),
        (2, "level2"),
        (3, "level3")
      )
    ).toDF("a", "label")
    val onlyDistinctLevelsDF: DataFrame = ss.sparkContext.parallelize(
      Seq(
        (1, "level1"),
        (2, "level2"),
        (3, "level2")
      )
    ).toDF("a", "label")

    val samples: Seq[(TemplateTree[TemplateMember], DataFrame)] = Seq(
      (LeafTemplate(DecisionTree()), threeDistinctLevelsDF),
      (LeafTemplate(DecisionTree()), onlyDistinctLevelsDF)
    )

    assertThrows[IllegalArgumentException] {
      SparkGenericBagging().checkThatWeHaveSameSetsOfCategoricalLevelsForAllSubmembers(samples)(ss)
    }
  }

  // We have the same test but in benchmarking
  ignore("Spark Bagging should calculate over multiple decision trees( Classification problem )") {

    val models = Seq(
      LeafTemplate(DecisionTree()), //TODO We need n-classes +2 base models to be able to find majority
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree()),
      LeafTemplate(DecisionTree())
    )

    val ensemb = NodeTemplate(SparkGenericBagging(), models)

    println(TemplateTreeHelper.renderAsString_v2(ensemb))

    val shufflingSeed = new Random().nextLong()
    println(s"Shuffling seed for current test: $shufflingSeed")

    val data = Datasets.getIrisDataFrame(shufflingSeed)

    val Array(trainingSplit, testSplit) = data.randomSplit(Array(0.67, 0.33), shufflingSeed)

    val problemType = ProblemType.MultiClassClassificationProblem

    val baggingF1 = ensemb.evaluateFitness(trainingSplit, testSplit, problemType, hpFieldFromCoevolution = Some(HyperParametersField.default), seed = shufflingSeed).getCorrespondingMetric

    val dtF1 = DecisionTree().fitnessError(trainingSplit, testSplit, problemType).getMetricByName("f1")

    println("Bagging's F1:" + baggingF1)
    println("DT's F1:" + dtF1)

    baggingF1 >= dtF1 should be(true) // This might not be the true all the time
  }

  test("Spark Bagging should calculate over complex tree algorithm( Classification problem )") {
    val metric = "f1"
    ConfigProvider.clearOverride.addOverride(
      s"""
        |evolution {
        |  hyperParameterDimension {
        |    enabled = false
        |  }
        |  evaluation {
        |    multiclass.metric = "$metric"
        |  }
        |}
      """)

    val models = Seq(
      LeafTemplate(DecisionTree()), //TODO We need n-classes +2 base models to be able to find majority
      LeafTemplate(Bayesian()),
      LeafTemplate(new LogisticRegressionModel())
    )

    val ensemb = NodeTemplate(SparkGenericBagging(), models)

    println(TemplateTreeHelper.renderAsString_v2(ensemb))

    val data = SparkMLUtils.loadResourceDF("/iris.csv")

    val preparedData = data
      .withColumnRenamed("fl_class", "label")
      .withColumn("uniqueIdColumn", monotonically_increasing_id)

    def basePredictorsFeaturesAssembler = new VectorAssembler()
      .setInputCols(Array("s_length", "s_width", "p_length", "p_width"))
      .setOutputCol("features")

    val featuresAssembled = basePredictorsFeaturesAssembler.transform(preparedData)

    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .setStringOrderType("alphabetAsc")
      .fit(featuresAssembled)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    logger.debug("StringIndexer: " + labelIndexer.labels.mkString(" | "))

    val preprocessedDF = featuresAssembled.applyTransformation(labelIndexer)

//    preprocessedDF.showN_AndContinue(100)

    val Array(trainingSplit, testSplit) = preprocessedDF.randomSplit(Array(0.67, 0.33), 11L)

    val problemType = ProblemType.MultiClassClassificationProblem

    val baggingF1 = ensemb.evaluateFitness(trainingSplit, testSplit, problemType, hpFieldFromCoevolution = Some(HyperParametersField.default)).getCorrespondingMetric

    val dtF1 = DecisionTree().fitnessError(trainingSplit, testSplit, problemType, hpFieldFromCoevolution = None).getMetricByName("f1")

    println("Bagging's F1:" + baggingF1)
    println("DT's F1:" + dtF1)

    baggingF1 should be(dtF1 +- 0.4) // This might not be the true all the time
  }

  test("probabilities are generated properly based on rawPredictions from base models") {
    val predictionFromA = 0.45048842525515953
    val predictionFromB = 0.4244651871844854
    val probabilityA = predictionFromA * 1 / (predictionFromA + predictionFromB)
    val probabilityB = predictionFromB * 1 / (predictionFromA + predictionFromB)

    probabilityA should be >= probabilityB
    probabilityA + probabilityB should be (1.0)
  }

  test("grouping") {
    val list = Seq(1,1,1,5)
    val weighted = (0.13 / 0.52) + (0.2 / 0.52)
    val res = list.zipWithIndex.groupBy(_._1).map{case (clazz, items) => (clazz, items.size.toDouble / list.size)}

    val ttt = 3737
  }


}


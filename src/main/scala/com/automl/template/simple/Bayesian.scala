package com.automl.template.simple

import com.automl.{ConfigProvider, ConsistencyChecker, LogLossCustom, PaddedLogging}
import com.automl.evolution.dimension.hparameter._
import com.automl.exception.SuspiciousPerformanceException
import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.spark.SparkSessionProvider
import com.automl.template.{EvaluationMagnet, TemplateMember}
import com.automl.teststrategy.{TestStrategy, TrainingTestSplitStrategy}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql._
import org.apache.spark.sql.types.{DoubleType, StringType}
import utils.SparkMLUtils

case class Bayesian(hpGroup: BayesianHPGroup = BayesianHPGroup())(implicit val logPaddingSize: Int = 0)
  extends SimpleModelMember
    with ClassificationMetricsHelper
    with SparkSessionProvider
    with PaddedLogging
    with ConsistencyChecker{

  override def name: String = "Bayesian " + super.name

  var hpGroupInternal: HyperParametersGroup[_ <: MutableHParameter[Double, _]] = hpGroup

  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case BinaryClassificationProblem => true
    case MultiClassClassificationProblem => true
    case RegressionProblem => true
  }
  import  SparkMLUtils._

  override val testStrategy: TestStrategy = new TrainingTestSplitStrategy()

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???


  override def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType, hpFieldFromCoevolution: Option[HyperParametersField]): FitnessResult = {

    debug(s"Evaluating $name ...")

// TODO we can use   model.numClasses

    problemType match {
      case RegressionProblem =>
//        val metadata = trainDF.schema.apply("label").dataType.isInstanceOf[StringType]
        val nb = new NaiveBayes()

        val model = nb.fit(trainDF)

        val predictions = model.transform(testDF)

        predictions.cache()

        val evaluator = new RegressionEvaluator()

        import SparkMLUtils._
        import ss.implicits._
        val rmse: Double = evaluator.evaluate(predictions.withColumnReplace("prediction", $"prediction".cast(DoubleType)))

        info(s"$name : RMSE = " + rmse)
        FitnessResult(Map("rmse" -> rmse), problemType, predictions.drop("rawPrediction").drop("probability"))

      case MultiClassClassificationProblem | BinaryClassificationProblem => //TODO generalize to a common method of evaluation for this type of problem.

        val config = ConfigProvider.config.getConfig("evolution")
        val validationStrategy = config.getString("templateDimension.validationStrategy")

        val classes = trainDF.select("indexedLabel").distinct().collect().map(_.getDouble(0))

        consistencyCheck {
          require(classes contains (0.0), s"Bayesian labels should have all indexes and be zero-based but instead: ${classes.mkString(",")}")
          //require(classes.length == 6, s"For glass dataset there should be 5 classes but instead: ${classes.mkString(",")}")
        }

        val nb = new NaiveBayes()
          .setModelType("multinomial")
          .setLabelCol("indexedLabel")

        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("indexedLabel")
          .setPredictionCol("prediction")
          .setMetricName("f1")

        val activeHPGroup: HyperParametersGroup[_] = getRelevantHPGroupFromActiveHPField(config, hpFieldFromCoevolution).getOrElse(hpGroup)


        if(validationStrategy == "cv") {
          val paramGrid = new ParamGridBuilder()
          val configuredParamGrid = activeHPGroup.hpParameters.foldLeft(paramGrid)((res, next) => next match {
            case p@Smoothing(_) =>
              debug(s"Bayesian's smoothing hyper-parameter was set to ${p.currentValue}")
              res.addGrid(nb.smoothing, Array(p.currentValue))
          }).build()
          val cv = new CrossValidator()
            .setEstimator(nb)
            .setEvaluator(evaluator)
            .setEstimatorParamMaps(configuredParamGrid)
            .setNumFolds(3)
            .setParallelism(2) // TODO 2 or ??

          val modelCV = cv.fit(trainDF)
          val f1CV = modelCV.avgMetrics(0) // <- this is averaged metric whereas `evaluator.setMetricName("f1").evaluate(predictions)` will return metric computed only on test data
          val predictions = modelCV.transform(testDF)

          //Unused
//          val metrics = new MulticlassMetrics(predictions.select("prediction", "indexedLabel").rdd.map(r => (r.getDouble(0), r.getDouble(1))))

          val logLoss = LogLossCustom.compute(predictions)

          printConfusionMatrix(predictions, testDF)

          val mapOfMetrics = Map("f1" -> f1CV, "accuracy" -> -1.0, "logloss" -> logLoss)
          info(s"Finished. $name : ${mapOfMetrics.map{nameToValue => nameToValue._1 + " = " + nameToValue._2}.mkString("",",", "")}. Number of rows = ${trainDF.count()} / ${testDF.count()}")
          FitnessResult(mapOfMetrics, problemType, predictions)
        } else {

          val naiveBayesWithHP = activeHPGroup.hpParameters.foldLeft(nb)((res, next) => next match {
            case p@Smoothing(_) =>
              debug(s"Bayesian smoothing hyper-parameter was set to ${p.currentValue}")
              res.setSmoothing(p.currentValue)
          })

          //        val paramGrid = new ParamGridBuilder()
          //          .addGrid(nb.smoothing, Array(1.0/*, 2.0, 3.0*/))
          //          .build()
          //
          //        val pipeline = new Pipeline()
          //          .setStages(Array(nb))
          //
          //        val evaluator = new MulticlassClassificationEvaluator()
          //          .setLabelCol("indexedLabel")
          //          .setPredictionCol("prediction")
          //
          //        val isLargerBetter = evaluator.isLargerBetter // TODO
          //
          //        val cv = new CrossValidator()
          //          .setEstimator(pipeline)
          //          .setEvaluator(evaluator)
          //          .setEstimatorParamMaps(paramGrid)
          //          .setNumFolds(3)   //best out of grid's parameters will be returned based on averaged over `setNumFolds` folds validation

          val model = naiveBayesWithHP.fit(trainDF)

          //        debug("Best Bayesian params: " + model.getEstimatorParamMaps.zip(model.avgMetrics).mkString(",").toString)

          val predictions = model.transform(testDF)

          val metrics = new MulticlassMetrics(predictions.select("prediction", "indexedLabel").rdd.map(r => (r.getDouble(0), r.getDouble(1))))

          //        MulticlassMetricsHelper.showStatistics(predictions)
          val logLoss = LogLossCustom.compute(predictions)

          if (metrics.weightedFMeasure <= 0.01)
            throw SuspiciousPerformanceException("Bayesian predictions are too low")

          info(s"Finished. $name : F1 metric = " + metrics.weightedFMeasure + s". Number of rows = ${trainDF.count()} / ${testDF.count()}")
          FitnessResult(
            Map("f1" -> metrics.weightedFMeasure, "weightedPrecision" -> metrics.weightedPrecision, "weightedRecall" -> metrics.weightedRecall, "logloss" -> logLoss),
            problemType,
            predictions
          )
        }
    }
  }
}


object MulticlassMetricsHelper extends LazyLogging {

  def showStatistics(predictionAndLabel: DataFrame) = {

    // Instantiate metrics object
    val metrics = new MulticlassMetrics(predictionAndLabel.select("prediction", "indexedLabel").rdd.map(r => (r.getDouble(0), r.getDouble(1))))

    // Confusion matrix
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    // Overall Statistics
    val accuracy = metrics.accuracy
    println("Summary Statistics")
    println(s"Accuracy = $accuracy")

    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      println(s"Precision($l) = " + metrics.precision(l))
    }

    // Recall by label
    labels.foreach { l =>
      println(s"Recall($l) = " + metrics.recall(l))
    }

    // False positive rate by label
    labels.foreach { l =>
      println(s"FPR($l) = " + metrics.falsePositiveRate(l))
    }

    // F-measure by label
    labels.foreach { l =>
      println(s"F1-Score($l) = " + metrics.fMeasure(l))
    }

    // Weighted stats
    println(s"Weighted precision: ${metrics.weightedPrecision}")
    println(s"Weighted recall: ${metrics.weightedRecall}")
    println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
    println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")
  }
}
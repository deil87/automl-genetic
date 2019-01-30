package com.automl.template.simple

import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.spark.SparkSessionProvider
import com.automl.template.EvaluationMagnet
import com.automl.teststrategy.{TestStrategy, TrainingTestSplitStrategy}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.sql._
import org.apache.spark.sql.types.{DoubleType, StringType}
import utils.SparkMLUtils

case class Bayesian() extends SimpleModelMember with SparkSessionProvider with LazyLogging{
  override def name: String = "Bayesian " + super.name

  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case BinaryClassificationProblem => true
    case MultiClassClassificationProblem => true
    case RegressionProblem => true
  }
  import  SparkMLUtils._

  override val testStrategy: TestStrategy = new TrainingTestSplitStrategy()

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???


  override def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType): FitnessResult = {

    logger.debug(s"Evaluating $name ...")

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

        logger.info(s"$name : RMSE = " + rmse)
        FitnessResult(Map("rmse" -> rmse), problemType, predictions.drop("rawPrediction").drop("probability"))

      case MultiClassClassificationProblem | BinaryClassificationProblem => //TODO generalize to a common method of evaluation for this type of problem.
//        val isStringResponse = trainDF.schema.apply("label").dataType.isInstanceOf[StringType]

        val nb = new NaiveBayes()
          .setLabelCol("indexedLabel")

        val pipeline = new Pipeline()
          .setStages(Array(nb))

        val model = pipeline.fit(trainDF)

        val predictions = model.transform(testDF)

        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("indexedLabel")
          .setPredictionCol("prediction")

        val f1: Double = evaluator
          .setMetricName("f1")
          .evaluate(predictions)

        logger.info(s"Finished. $name : F1 metric = " + f1)
        FitnessResult(Map("f1" -> f1), problemType, predictions)

    }
  }
}

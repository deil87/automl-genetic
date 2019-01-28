package com.automl.template.simple

import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.spark.SparkSessionProvider
import com.automl.template.EvaluationMagnet
import com.automl.teststrategy.{TestStrategy, TrainingTestSplitStrategy}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.sql._
import org.apache.spark.sql.types.DoubleType
import utils.SparkMLUtils

case class Bayesian() extends SimpleModelMember with SparkSessionProvider with LazyLogging{
  override def name: String = "Bayesian " + super.name

  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case BinaryClassificationProblem => true
    case MultiClassClassificationProblem => true
    case RegressionProblem => true
  }

  override val testStrategy: TestStrategy = new TrainingTestSplitStrategy()

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???


  override def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType): FitnessResult = {

    logger.debug(s"Evaluating $name ...")
    val nb = new NaiveBayes()


    val model = nb.fit(trainDF)

    val predictions = model.transform(testDF)

    predictions.cache()

// TODO we can use   model.numClasses

    problemType match {
      case RegressionProblem =>
        val evaluator = new RegressionEvaluator()

        import SparkMLUtils._
        import ss.implicits._
        val rmse: Double = evaluator.evaluate(predictions.withColumnReplace("prediction", $"prediction".cast(DoubleType)))

        logger.info(s"$name : RMSE = " + rmse)
        FitnessResult(rmse, predictions.drop("rawPrediction").drop("probability"))

      case MultiClassClassificationProblem | BinaryClassificationProblem => //TODO generalize to a common method of evaluation for this type of problem.
        val evaluator = new MulticlassClassificationEvaluator() // What is binary?

        val f1: Double = evaluator.setMetricName("f1").evaluate(predictions)

        logger.info(s"Finished. $name : F1 metric = " + f1)
        FitnessResult(f1, predictions)

    }
  }
}

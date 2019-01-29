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
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.sql._
import org.apache.spark.sql.types.DoubleType
import utils.SparkMLUtils

case class DecisionTree() extends SimpleModelMember with SparkSessionProvider with LazyLogging{
  override def name: String = "DecisionTree " + super.name

  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case BinaryClassificationProblem => true
    case MultiClassClassificationProblem => true
    case RegressionProblem => true
  }

  override def testStrategy: TestStrategy = new TrainingTestSplitStrategy()


  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = null


  override def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType): FitnessResult = {

    logger.debug(s"Started evaluating $name ...")
    import  SparkMLUtils._

    import ss.implicits._

    val dtr = new DecisionTreeRegressor()
    val model = dtr.fit(trainDF)
    val predictions = model.transform(testDF)
     problemType match {
       case RegressionProblem =>
         val evaluator = new RegressionEvaluator()

         val rmse: Double = evaluator.evaluate(predictions)

         logger.info(s"Finished. $name : RMSE = " + rmse)
         FitnessResult(Map("rmse" -> rmse), problemType, predictions)
       case MultiClassClassificationProblem | BinaryClassificationProblem =>
         val evaluator = new MulticlassClassificationEvaluator() // What is binary?

         val f1: Double = evaluator.setMetricName("f1").evaluate(predictions)

         logger.info(s"Finished. $name : F1 metric = " + f1)
         FitnessResult(Map("f1" -> f1), problemType, predictions)
     }

  }
}

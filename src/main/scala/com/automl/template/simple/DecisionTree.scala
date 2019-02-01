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
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, NaiveBayes}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.sql._
import org.apache.spark.sql.types.DoubleType
import utils.SparkMLUtils

import scala.util.Random

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

     problemType match {
       case RegressionProblem =>
         val dtr = new DecisionTreeRegressor()
         val model = dtr.fit(trainDF)
         val predictions = model.transform(testDF)

         val evaluator = new RegressionEvaluator()

         val rmse: Double = evaluator.evaluate(predictions)

         logger.info(s"Finished. $name : RMSE = " + rmse)
         FitnessResult(Map("rmse" -> rmse), problemType, predictions)
       case MultiClassClassificationProblem | BinaryClassificationProblem =>

         val maxDepth = Random.nextInt(7) + 5
         val dtr = new DecisionTreeClassifier()
           .setMaxDepth(maxDepth)
           .setLabelCol("indexedLabel")

         val pipeline = new Pipeline()
           .setStages(Array(dtr))

         val model = pipeline.fit(trainDF)

         val predictions = model.transform(testDF)

         val evaluator = new MulticlassClassificationEvaluator()
           .setLabelCol("indexedLabel")
           .setPredictionCol("prediction")

         val f1: Double = evaluator.setMetricName("f1").evaluate(predictions)
         val accuracy: Double = evaluator.setMetricName("accuracy").evaluate(predictions)

         val indexOfStageForModelInPipeline = 0
         val treeModel = model.stages(indexOfStageForModelInPipeline).asInstanceOf[DecisionTreeClassificationModel]
         logger.debug("Learned classification tree model:\n" + treeModel.toDebugString)

         logger.info(s"Finished. $name : F1 metric = " + f1)
         FitnessResult(Map("f1" -> f1, "accuracy" -> accuracy), problemType, predictions)
     }

  }
}

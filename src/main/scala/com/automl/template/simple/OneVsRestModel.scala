package com.automl.template.simple

import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.EvaluationMagnet
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql._


//TODO compare with just LogisticRegressionModel class
case class OneVsRestModel() extends SimpleModelMember with LazyLogging{
  override def name: String = "OneVsRest " + super.name

  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case BinaryClassificationProblem => true
    case MultiClassClassificationProblem => true
    case RegressionProblem => false
  }

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = null

  override def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType): FitnessResult = {

    problemType match {
      case MultiClassClassificationProblem | BinaryClassificationProblem=>
        val classifier = new LogisticRegression()
          .setMaxIter(10)
          .setTol(1E-6)
          .setFitIntercept(true)

        // instantiate the One Vs Rest Classifier.
        val ovr = new OneVsRest().setClassifier(classifier)

        // train the multiclass model.
        val ovrModel = ovr.fit(trainDF)

        // score the model on test data.
        val predictions = ovrModel.transform(testDF)

        // obtain evaluator.
        val evaluator = new MulticlassClassificationEvaluator()
          .setMetricName("accuracy")

        // compute the classification error on test data.
        val accuracy = evaluator.evaluate(predictions)
        FitnessResult(???, ???, ???)
    }
  }

}
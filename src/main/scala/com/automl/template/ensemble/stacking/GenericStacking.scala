package com.automl.template.ensemble.stacking

import com.automl.{LogLossCustom, PaddedLogging}
import com.automl.helper.FitnessResult
import com.automl.regressor.EnsemblingRegressor
import com.automl.template.{EvaluationMagnet, TemplateMember, TemplateTree, TreeContext}
import com.automl.template.ensemble.EnsemblingModelMember
import com.automl.classifier.ensemble.stacking.SparkGenericStacking
import com.automl.evolution.dimension.hparameter.{HyperParametersField, HyperParametersGroup, MutableHParameter, StackingHPGroup}
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.{Pipeline, PipelineStage, Predictor}
import org.apache.spark.sql.DataFrame
import utils.SparkMLUtils._


case class GenericStacking(unusedMetaLearner: PipelineStage = new LinearRegression(),
                           hpg: StackingHPGroup = StackingHPGroup()
                          )(implicit val logPaddingSize: Int = 0)
  extends StackingMember with PaddedLogging {
  override def name: String = "SparkStacking " + super.name

  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case BinaryClassificationProblem => ??? // TODO enable
    case MultiClassClassificationProblem => true
    case RegressionProblem => ??? // TODO enable
  }

  var hpGroupInternal: HyperParametersGroup[_ <: MutableHParameter[Double, _]] = hpg

  override def ensemblingFitnessError[A <: TemplateMember](trainDF: DataFrame,
                                                           testDF: DataFrame,
                                                           subMembers: Seq[TemplateTree[A]],
                                                           problemType: ProblemType,
                                                           hyperParamsField: Option[HyperParametersField],
                                                           seed: Long)
                                                          (implicit tc: TreeContext = TreeContext()): FitnessResult = {
    val stackingNumberOfFolds = 3
    logger.debug(s"Evaluating $name ...")
    val responseColumnName = if(problemType.isClassification) "indexedLabel" else "label"
    val stacking = new SparkGenericStacking(stackingNumberOfFolds, responseColumnName)

    logger.debug(s"Stacking number of folds is set to $stackingNumberOfFolds")
    stacking.foldingStage(trainDF, testDF)

    generateTrainingDataForSubMembers(trainDF, subMembers, hyperParamsField, seed)
      .foldLeft(stacking) {
        case (stackingModel, (nextMember, trainingSample)) =>
          stackingModel.addModel(nextMember, trainingSample, testDF, problemType, hyperParamsField)
      }

    logger.debug(s"All Stacking submembers were added to a Stacking model")

    problemType match {
      case RegressionProblem =>
        val finalPredictions = stacking.performStacking(unusedMetaLearner)
          .select("uniqueIdColumn", "features", "prediction") //TODO make sure that performStacking is returning predictions for testDF

        val evaluator = new RegressionEvaluator()

        val predictionsReunitedWithLabels = finalPredictions.join(testDF.select("label", "uniqueIdColumn"), "uniqueIdColumn")

        val rmse = evaluator.evaluate(predictionsReunitedWithLabels)
        logger.info(s"Finished. $name RMSE Final:" + rmse)
        FitnessResult(Map("rmse" -> rmse), problemType, predictionsReunitedWithLabels)

      case MultiClassClassificationProblem | BinaryClassificationProblem =>
        import testDF.sparkSession.implicits._
        import org.apache.spark.sql.functions.rint

//        val metaLearner = new LinearRegression().setFeaturesCol("features").setLabelCol("indexedLabel")
        val metaLearner = new LogisticRegression()
//          .setWeightCol("weight") //it controls how much faith to put into particular training instance. We need kind of weights for the features
          .setFeaturesCol("features")
          .setLabelCol("indexedLabel")
          .setMaxIter(20)

        logger.debug(s"Running performStacking() method")
        val stackedFrame = stacking.performStacking(metaLearner)
        logger.debug(s"Finished performStacking() method")

        val finalPredictions = stackedFrame
//          .showN_AndContinue(100, "Before rounding predictions from GenericStacking's metalearner")
//          .withColumnReplace("prediction", rint($"prediction")) //NOTE we need to round because LinearRegression metalearner returns raw predictions
//          .showN_AndContinue(500, "predictions from GenericStacking's metalearner")
          .select("uniqueIdColumn", "features", "prediction", "probability") //TODO make sure that performStacking is returning predictions for testDF
          .cache()

        val predictionsReunitedWithLabels = finalPredictions.join(testDF.select("indexedLabel", "uniqueIdColumn"), "uniqueIdColumn")
//          .showN_AndContinue(500, "GenericStacking predictions")
          .cache()

        val evaluator = new MulticlassClassificationEvaluator()
          .setPredictionCol("prediction") //Prediction from Stacking
          .setLabelCol("indexedLabel") // True labels from testDF
          .setMetricName("f1")

        // Based on `probability` column that is coming from metaLearner ( in current case LogisticRegression)
        val logLoss = LogLossCustom.compute(predictionsReunitedWithLabels)

        val f1 = evaluator.evaluate(predictionsReunitedWithLabels)
        logger.info(s"Finished $name : F1 = $f1, logloss = $logLoss")
        FitnessResult(Map("f1" -> f1, "logloss" -> logLoss), problemType, predictionsReunitedWithLabels)
    }
  }

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

}


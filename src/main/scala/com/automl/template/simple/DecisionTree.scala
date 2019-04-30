package com.automl.template.simple

import com.automl.{ConfigProvider, PaddedLogging}
import com.automl.evolution.dimension.hparameter.{DecisionTreeHPGroup, MaxDepth}
import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.spark.SparkSessionProvider
import com.automl.template.EvaluationMagnet
import com.automl.teststrategy.{TestStrategy, TrainingTestSplitStrategy}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, NaiveBayes}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql._
import utils.SparkMLUtils


case class DecisionTree(hpGroup: DecisionTreeHPGroup = DecisionTreeHPGroup.default)(implicit val logPaddingSize: Int = 0)
  extends SimpleModelMember
  with SparkSessionProvider with PaddedLogging{

  override def name: String = "DecisionTree " + super.name

  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case BinaryClassificationProblem => true
    case MultiClassClassificationProblem => true
    case RegressionProblem => true
  }

  override def testStrategy: TestStrategy = new TrainingTestSplitStrategy()


  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = null


  override def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType): FitnessResult = {

    trainDF.cache()
    testDF.cache()
    debug(s"Started evaluating $name ...")
    import  SparkMLUtils._

    import ss.implicits._

     problemType match {
       case RegressionProblem =>
         val dtr = new DecisionTreeRegressor()
         val model = dtr.fit(trainDF)
         val predictions = model.transform(testDF)

         val evaluator = new RegressionEvaluator()

         val rmse: Double = evaluator.evaluate(predictions)

         info(s"Finished. $name : RMSE = " + rmse)
         FitnessResult(Map("rmse" -> rmse), problemType, predictions)
       case MultiClassClassificationProblem | BinaryClassificationProblem =>

         val dtr = new DecisionTreeClassifier()
           .setLabelCol("indexedLabel")

         //Temporarily using RandomGridSearch for hyper parameters to prove our assumtion that ensembles of models are better than simple models.
         val config = ConfigProvider.config.getConfig("evolution")
         val performGridSearch = config.getBoolean("hpGridSearch")
         val hpCoevolutionIsEnabled = config.getBoolean("hyperParameterDimension.enabled")

         val evaluator = new MulticlassClassificationEvaluator()
           .setLabelCol("indexedLabel")
           .setPredictionCol("prediction")
           .setMetricName("f1")



         val model = if(performGridSearch) {
           val paramGrid = new ParamGridBuilder()
             .addGrid(dtr.maxDepth, Array(3, 5, 7, 10))
//             .addGrid(dtr.maxBins, Array(32, 48, 64))
             .build()

           val cv = new CrossValidator()
             .setEstimator(dtr)
             .setEvaluator(evaluator)
             .setEstimatorParamMaps(paramGrid)
             .setNumFolds(2)

           cv.fit(trainDF)

         } else {
           val classifier = if (hpCoevolutionIsEnabled) {
             hpGroup.hpParameters.foldLeft(dtr)((res, next) => next match {
               case p@MaxDepth(_) =>
                 debug(s"DecisionTree max_depth hyper-parameter was set to ${p.currentValue}")
                 res.setMaxDepth(p.currentValue.toInt)
             })
           } else dtr

           val pipeline = new Pipeline()
             .setStages(Array(classifier))


           pipeline.fit(trainDF)
         }

//         debug("Learned classification tree model:\n" + model.asInstanceOf[CrossValidatorModel].bestModel.asInstanceOf[DecisionTreeClassificationModel].toDebugString)
//         debug("Learned classification tree model:\n" + model.asInstanceOf[PipelineModel].stages(0).asInstanceOf[DecisionTreeClassificationModel].toDebugString)


         val predictions = model.transform(testDF)

         val f1: Double = evaluator.setMetricName("f1").evaluate(predictions)
         val accuracy: Double = evaluator.setMetricName("accuracy").evaluate(predictions)

//         val indexOfStageForModelInPipeline = 0
//         val treeModel = model.stages(indexOfStageForModelInPipeline).asInstanceOf[DecisionTreeClassificationModel]
//         debug("Learned classification tree model:\n" + treeModel.toDebugString)

         info(s"Finished. $name : F1 metric = " + f1 + s". Number of rows = ${trainDF.count()} / ${testDF.count()}")
         FitnessResult(Map("f1" -> f1, "accuracy" -> accuracy), problemType, predictions)
     }

  }
}

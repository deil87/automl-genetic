package com.automl.template.simple

import com.automl.{ConfigProvider, LogLossCustom, PaddedLogging}
import com.automl.evolution.dimension.hparameter._
import com.automl.helper.{FitnessResult, TemplateTreeHelper}
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.spark.SparkSessionProvider
import com.automl.template.EvaluationMagnet
import com.automl.teststrategy.{TestStrategy, TrainingTestSplitStrategy}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql._
import utils.SparkMLUtils

import scala.util.Random


case class DecisionTree(hpGroup: DecisionTreeHPGroup = DecisionTreeHPGroup(),
                        seed: Long = Random.nextLong()
                       )(implicit val logPaddingSize: Int = 0)
  extends SimpleModelMember
    with ClassificationMetricsHelper
    with SparkSessionProvider
    with PaddedLogging{

  override def name: String = "DecisionTree " + super.name

  var hpGroupInternal: HyperParametersGroup[_ <: MutableHParameter[Double, _]] = hpGroup

  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case BinaryClassificationProblem => true
    case MultiClassClassificationProblem => true
    case RegressionProblem => true
  }

  override def testStrategy: TestStrategy = new TrainingTestSplitStrategy()


  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = null


  override def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType, hpFieldFromCoevolution: Option[HyperParametersField]): FitnessResult = {

    trainDF.cache()
    testDF.cache()
    info(s"Started evaluating $name ...")
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
         val validationStrategy = config.getString("templateDimension.validationStrategy")
         val hpCoevolutionIsEnabled = config.getBoolean("hyperParameterDimension.enabled")

         val evaluator = new MulticlassClassificationEvaluator()
           .setLabelCol("indexedLabel")
           .setPredictionCol("prediction")
           .setMetricName("f1")

         debug(s"DecisionTree is being run with performHPGridSearch=$performGridSearch and validationStrategy=$validationStrategy")

         if(performGridSearch) {
           val paramGrid = new ParamGridBuilder()
             .addGrid(dtr.maxDepth, Array(3, 4, 5, 7))
//             .addGrid(dtr.maxBins, Array(32, 48, 64))
             .build()

           val cv = new CrossValidator()
             .setEstimator(dtr)
             .setEvaluator(evaluator)
             .setEstimatorParamMaps(paramGrid)
             .setNumFolds(3)

           val modelGSCV= cv.fit(trainDF)
           val predictions = modelGSCV.transform(testDF)

//           predictions.showAll()

           val f1: Double = evaluator.setMetricName("f1").evaluate(predictions)
           val accuracy: Double = evaluator.setMetricName("accuracy").evaluate(predictions)

           val logLoss = LogLossCustom.compute(predictions)

           info(s"Finished. $name : F1 metric = " + f1 + s". Number of rows = ${trainDF.count()} / ${testDF.count()}")
           FitnessResult(Map("f1" -> f1, "accuracy" -> accuracy, "logloss" -> logLoss), problemType, predictions)

         } else { // non-HPGridSearch version

           // could be also moved to TemplateMember
           val activeHPGroup: HyperParametersGroup[_] = getRelevantHPGroupFromActiveHPField(hpFieldFromCoevolution).getOrElse(hpGroup)

           // We can't train CV on `train+test` data and then predict on itself -> overfitted resuls.
           // We need at least `test` split  to get predictions which could be used to find phenotypic similarity.
           // But we can use CV averaged estimate instead of estimate on test split
           if(validationStrategy == "cv") {
             val paramGrid = new ParamGridBuilder()
             val configuredParamGrid = activeHPGroup.hpParameters.foldLeft(paramGrid)((res, next) => next match {
               case p@MaxDepth(_) =>
                 debug(s"DecisionTree's max_depth hyper-parameter was set to ${p.currentValue}")
                 res.addGrid(dtr.maxDepth, Array(p.currentValue.toInt))
             }).build()
             val cv = new CrossValidator()
               .setEstimator(dtr)
               .setEvaluator(evaluator)
               .setEstimatorParamMaps(configuredParamGrid)
               .setNumFolds(3)
               .setParallelism(2) // TODO 2 or ??
               .setSeed(seed)

             val modelCV = cv.fit(trainDF) // TODO maybe we need to make testDF to be optional and used trainingDF as CV
             val f1CV = modelCV.avgMetrics(0) // <- this is averaged metric whereas `evaluator.setMetricName("f1").evaluate(predictions)` will return metric computed only on test data
             val predictions = modelCV.transform(testDF)

             //Unused
             val f1 = evaluator.setMetricName("f1").evaluate(predictions)

             val logLoss = LogLossCustom.compute(predictions)

             printConfusionMatrix(predictions, testDF)
             //        MulticlassMetricsHelper.showStatistics(predictions)
             val mapOfMetrics = Map("f1" -> f1, "logloss" -> logLoss)

             info(s"Finished. $name ${activeHPGroup.hpParameters.mkString(",")} : ${mapOfMetrics.map{ nameToValue => nameToValue._1 + " = " + nameToValue._2}.mkString("",",", "")}. Number of rows = train:${trainDF.count()} / test:${testDF.count()}")

             FitnessResult(Map("f1" -> f1CV, "accuracy" -> -1, "logloss" -> logLoss), problemType, predictions)
           } else {
             val classifier = activeHPGroup.hpParameters.foldLeft(dtr)((res, next) => next match {
               case p@MaxDepth(_) =>
                 debug(s"DecisionTree max_depth hyper-parameter was set to ${p.currentValue}")
                 res.setMaxDepth(p.currentValue.toInt)
             })
             val pipeline = new Pipeline()
               .setStages(Array(classifier))

             val modelTT = pipeline.fit(trainDF)

             val predictions = modelTT.transform(testDF)

             val f1: Double = evaluator.setMetricName("f1").evaluate(predictions)
             val accuracy: Double = evaluator.setMetricName("accuracy").evaluate(predictions)
             val logLoss = LogLossCustom.compute(predictions)

             //         val indexOfStageForModelInPipeline = 0
             //         val treeModel = model.stages(indexOfStageForModelInPipeline).asInstanceOf[DecisionTreeClassificationModel]
             //         debug("Learned classification tree model:\n" + treeModel.toDebugString)

             val mapOfMetrics = Map("f1" -> f1, "accuracy" -> accuracy, "logloss" -> logLoss)
             info(s"Finished. ${name} ${activeHPGroup.hpParameters.mkString(",")} : ${mapOfMetrics.map{ nameToValue => nameToValue._1 + " = " + nameToValue._2}.mkString("",",", "")}. Number of rows = train:${trainDF.count()} / test:${testDF.count()}")
             FitnessResult(mapOfMetrics, problemType, predictions)
           }
         }

//         debug("Learned classification tree model:\n" + model.asInstanceOf[CrossValidatorModel].bestModel.asInstanceOf[DecisionTreeClassificationModel].toDebugString)
//         debug("Learned classification tree model:\n" + model.asInstanceOf[PipelineModel].stages(0).asInstanceOf[DecisionTreeClassificationModel].toDebugString)

     }

  }
}

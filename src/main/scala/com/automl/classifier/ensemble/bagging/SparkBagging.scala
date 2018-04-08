package com.automl.classifier.ensemble.bagging

import com.automl.helper.FitnessResult
import com.automl.template.{TemplateMember, TemplateTree, TreeContext}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import utils.SparkMLUtils


class SparkBagging[A <: TemplateMember](models: Seq[TemplateTree[A]]) {

  import utils.SparkMLUtils._
  def fitnessError(trainingDF: DataFrame, testDF: DataFrame)(implicit tc: TreeContext = TreeContext()): FitnessResult = {

    val results: Seq[(TemplateTree[A], FitnessResult)] = models.zipWithIndex.map{ case (model, modelIdx) =>
      val trainingSample = trainingDF.sample(withReplacement = false, 0.6)

      //TODO  Should we keep aside testDF? Maybe we are computing just training error. We need to split trainingSample into (train,test)
      (model, model.evaluateFitness(trainingSample, testDF))
    }
    val dfWithPredictionsFromBaseModels: Seq[DataFrame] = results
      .map(_._2.dfWithPredictions)

    import SparkMLUtils._
    val unionedPredictions: Dataset[Row] =
      dfWithPredictionsFromBaseModels
      .reduce((a, b) => {
        val tmp = results
        val ordered: Dataset[Row] = b.select(a.columns.head, a.columns.tail: _*)
        val res = a.union(ordered)
        res
      }).cache()

    val mergedAndRegressedDF = unionedPredictions
      .groupBy("uniqueIdColumn") // Or we should join aggregation results with other missing columns
      .avg("prediction")
//      .showN_AndContinue(10)
//      .showCount_AndContinue()
      .withColumnRenamed("avg(prediction)", "prediction") // TODO using avg regressor for now
      .join(dfWithPredictionsFromBaseModels.head.drop("prediction"), Seq("uniqueIdColumn"), joinType="left_outer" )
//      .showN_AndContinue(10)
//      .showCount_AndContinue()
      .cache()

    results.foreach(_._2.dfWithPredictions.unpersist())

    val evaluator = new RegressionEvaluator()

    val rmse: Double = evaluator.evaluate(mergedAndRegressedDF)

    FitnessResult(rmse, mergedAndRegressedDF)
  }

}
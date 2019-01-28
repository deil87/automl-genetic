package com.automl.classifier.ensemble.bagging

import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.regressor.EnsemblingRegressor
import com.automl.template.ensemble.EnsemblingModelMember
import com.automl.template.{TemplateMember, TemplateTree, TreeContext}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import utils.SparkMLUtils


class SparkBagging[A <: TemplateMember]() extends EnsemblingModelMember{

  import utils.SparkMLUtils._

  override def ensemblingRegressor: EnsemblingRegressor = ???

  override def ensemblingFitnessError[A <: TemplateMember](trainDF: DataFrame, testDF: DataFrame, subMembers: Seq[TemplateTree[A]], problemType: ProblemType)
                                                          (implicit tc: TreeContext): FitnessResult = {

    val results: Seq[(TemplateTree[A], FitnessResult)] = subMembers.zipWithIndex.map{ case (model, modelIdx) =>
      val trainingSample = trainDF.sample(withReplacement = false, 0.6)

      //TODO  Should we keep aside testDF? Maybe we are computing just training error. We need to split trainingSample into (train,test)
      (model, model.evaluateFitness(trainingSample, testDF, problemType))
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
      .withColumnRenamed("avg(prediction)", "prediction") // TODO using avg regressor for now
      .join(dfWithPredictionsFromBaseModels.head.drop("prediction"), Seq("uniqueIdColumn"), joinType="left_outer" )
      .cache()

    results.foreach(_._2.dfWithPredictions.unpersist())

    val evaluator = new RegressionEvaluator()

    val rmse: Double = evaluator.evaluate(mergedAndRegressedDF)

    FitnessResult(rmse, mergedAndRegressedDF)
  }

}

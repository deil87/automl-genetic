package com.automl.classifier.ensemble.bagging

import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.regressor.{AverageRegressor, EnsemblingRegressor, MajorityVoteRegressor}
import com.automl.template.ensemble.EnsemblingModelMember
import com.automl.template.ensemble.bagging.BaggingMember
import com.automl.template.{TemplateMember, TemplateTree, TreeContext}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql._
import utils.SparkMLUtils
import org.apache.spark.ml.linalg.{Vector => MLVector}

import scala.collection.mutable

//TODO consider moving closer to BaggingMember
case class SparkGenericBagging() extends BaggingMember with LazyLogging{

  import utils.SparkMLUtils._

  override def ensemblingRegressor[_](problemType: ProblemType): EnsemblingRegressor[_] = problemType match {
    case RegressionProblem => new AverageRegressor()
    case MultiClassClassificationProblem | BinaryClassificationProblem => new MajorityVoteRegressor()
  }

  override def ensemblingFitnessError[A <: TemplateMember](trainDF: DataFrame,
                                                           testDF: DataFrame,
                                                           subMembers: Seq[TemplateTree[A]],
                                                           problemType: ProblemType,
                                                           hyperParamsMap: HyperParametersField)
                                                          (implicit tc: TreeContext): FitnessResult = {

    import trainDF.sparkSession.implicits._

    val trainingSamplesForSubmembers = subMembers.map { model =>
      (model, trainDF.sample(withReplacement = false, 0.8)) // How to sample?
    }

    val results: Seq[(TemplateTree[A], FitnessResult)] = trainingSamplesForSubmembers.zipWithIndex.map{ case ((model, trainingSample), modelIdx) =>

      //TODO  Should we keep aside testDF? Maybe we are computing just training error. We need to split trainingSample into (train,test)
      //TODO We need to make sure that Indexers of submembers have the same sets of indexes ( sorting frame by `label`).
      // But by sampling trainDF we can not guarantee that we will not miss some level
      (model, model.evaluateFitness(trainingSample, testDF, problemType, hyperParamsMap))
    }

    val dfWithPredictionsFromBaseModels: Seq[DataFrame] = results
      .map(_._2.dfWithPredictions)

    import SparkMLUtils._

    val joinedPredictions: Dataset[Row] =
      dfWithPredictionsFromBaseModels
//        .map(_.showN_AndContinue(20))
        .zipWithIndex
        .map{case (df, idx) =>
          df.select($"uniqueIdColumn", $"prediction")
            .withColumnRenamed("prediction", s"prediction$idx")
        }
        .reduce((a, b) => {
          a.join(b, "uniqueIdColumn")
        }).cache()
//        .showN_AndContinue(10, "After joining predictions")

    def generateMajorityVoteUDF = {
      import org.apache.spark.sql.functions.udf
      udf { v: MLVector =>
        val occurrences = mutable.Map.empty[Double, Int]
        v.toArray.foreach { x =>
          occurrences.update(x, occurrences.getOrElse(x, 0) + 1)
        }
        occurrences.toArray.sortWith(_._2 > _._2).head._1
      }
    }

    problemType match {
      case MultiClassClassificationProblem | BinaryClassificationProblem =>

        checkThatWeHaveSameSetsOfCategoricalLevelsForAllSubmembers(trainingSamplesForSubmembers)(trainDF.sparkSession)

        import org.apache.spark.sql.functions.col
        val predictionColumns: Array[Column] = dfWithPredictionsFromBaseModels.indices.toArray.map{ idx => col(s"prediction$idx")}

        def predictionsAssembler = new VectorAssembler()
          .setInputCols(predictionColumns.map(_.toString))
          .setOutputCol("base_models_predictions")

        val mergedAndRegressedDF =
            joinedPredictions
              .applyTransformation(predictionsAssembler)
//              .showN_AndContinue(10)
              .withColumn("majority_prediction", generateMajorityVoteUDF($"base_models_predictions"))
              .withColumnRenamed("majority_prediction", "prediction")
              //            .drop(predictionColumns.map(_.toString): _*)
              .join(dfWithPredictionsFromBaseModels.head.drop("prediction"), Seq("uniqueIdColumn"), joinType = "left_outer")
              .cache()

//        mergedAndRegressedDF.select(predictionColumns :+ $"indexedLabel" :+ $"prediction" :_*).showN_AndContinue(100)

        results.foreach(_._2.dfWithPredictions.unpersist()) //TODO doubling see below

        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("indexedLabel")
          .setPredictionCol("prediction")
          .setMetricName("f1")

        val f1 = evaluator
          .evaluate(mergedAndRegressedDF)

        logger.info(s"$name : f1 = " + f1)
        FitnessResult(Map("f1" -> f1), problemType, mergedAndRegressedDF)

      case RegressionProblem =>

        //TODO see how it is done in MultiClassClassificationProblem case. Maybe we need to combine vector of predictions.
        //TODO we need to use here `ensemblingRegressor` - maybe overload multiple averaging methods. Maybe put this 3 lines into there?
        import org.apache.spark.sql.functions.{col, lit}

        val predictionColumns: Array[Column] = dfWithPredictionsFromBaseModels.indices.toArray.map{ idx => col(s"prediction$idx")}

        val averageFunc: Column = predictionColumns.foldLeft(lit(0)) { (x, y) => x + y } / predictionColumns.length

        val mergedAndRegressedDF = joinedPredictions
//          .showN_AndContinue(100)
          .withColumn("avg(prediction)", averageFunc)
          .withColumnRenamed("avg(prediction)", "prediction")
          .drop(predictionColumns.map(_.toString):_*)
          .join(dfWithPredictionsFromBaseModels.head.drop("prediction"), Seq("uniqueIdColumn"), joinType="left_outer" )
          .cache()
//            .showN_AndContinue(100)

        results.foreach(_._2.dfWithPredictions.unpersist())

        val evaluator = new RegressionEvaluator()

        val rmse: Double = evaluator.evaluate(mergedAndRegressedDF)

        FitnessResult(Map("rmse" -> rmse), problemType, mergedAndRegressedDF)

    }

  }

  private[bagging] def checkThatWeHaveSameSetsOfCategoricalLevelsForAllSubmembers[A <: TemplateMember](trainingSamplesForSubmembers: Seq[(TemplateTree[A], DataFrame)])
                                                                                                      (ss: SparkSession):Unit = {
    import ss.implicits._
    val subsetsOfLevelsForAllSamples = trainingSamplesForSubmembers
      .map{sample =>
        val originalSample = sample._2
        val distinctLevels = originalSample.select("indexedLabel").distinct()
        distinctLevels.map(_.get(0).asInstanceOf[Double]).collect().toSet
      }

    val head = subsetsOfLevelsForAllSamples.head
    require(subsetsOfLevelsForAllSamples.tail.forall(set => head == set), "All samples for submembers should have the same level values")
  }

}

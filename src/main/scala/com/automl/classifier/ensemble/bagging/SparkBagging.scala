package com.automl.classifier.ensemble.bagging

import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.regressor.{AverageRegressor, EnsemblingRegressor, MajorityVoteRegressor}
import com.automl.template.ensemble.EnsemblingModelMember
import com.automl.template.ensemble.bagging.BaggingMember
import com.automl.template.{TemplateMember, TemplateTree, TreeContext}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row}
import utils.SparkMLUtils

import scala.collection.mutable

//TODO consider moving closer to BaggingMember
case class SparkBagging[A <: TemplateMember]() extends BaggingMember with LazyLogging{

  import utils.SparkMLUtils._

  override def ensemblingRegressor[_](problemType: ProblemType): EnsemblingRegressor[_] = problemType match {
    case RegressionProblem => new AverageRegressor()
    case MultiClassClassificationProblem | BinaryClassificationProblem => new MajorityVoteRegressor()
  }

  override def ensemblingFitnessError[A <: TemplateMember](trainDF: DataFrame,
                                                           testDF: DataFrame,
                                                           subMembers: Seq[TemplateTree[A]],
                                                           problemType: ProblemType)
                                                          (implicit tc: TreeContext): FitnessResult = {

    import trainDF.sparkSession.implicits._

    val trainingSamplesForSubmembers = subMembers.map { model =>
      (model, trainDF.sample(withReplacement = false, 0.8)) // How to sample?
    }

    def checkThatWeHaveSameSetsOfCatigoricalLevelsForAllSubmembers = {
      //Just to check
      val subsetsOfLevelsForAllSamples = trainingSamplesForSubmembers
        .map(_._2.select("label").distinct().map(_.get(0).asInstanceOf[String]).collect().toSet)

      val head = subsetsOfLevelsForAllSamples.head
      require(subsetsOfLevelsForAllSamples.tail.forall(set => head == set), "All samples for submembers should have the same level values")
    }

    checkThatWeHaveSameSetsOfCatigoricalLevelsForAllSubmembers

    //TODO !!!!!!!!!!!!!  support variable number of submembers in udf function. Check that indexes are the same in all submembers

    val results: Seq[(TemplateTree[A], FitnessResult)] = trainingSamplesForSubmembers.zipWithIndex.map{ case ((model, trainingSample), modelIdx) =>

      //TODO  Should we keep aside testDF? Maybe we are computing just training error. We need to split trainingSample into (train,test)
      //TODO We need to make sure that Indexers of submembers have the same sets of indexes ( sorting frame by `label`).
      // But by sampling trainDF we can not guarantee that we will not miss some level
      (model, model.evaluateFitness(trainingSample, testDF, problemType))
    }

    val dfWithPredictionsFromBaseModels: Seq[DataFrame] = results
      .map(_._2.dfWithPredictions)

    import SparkMLUtils._

    val joinedPredictions: Dataset[Row] =
      dfWithPredictionsFromBaseModels
        .map(_.showN_AndContinue(10))
        .zipWithIndex
        .map{case (df, idx) =>
          df.select($"uniqueIdColumn", $"prediction")
            .withColumnRenamed("prediction", s"prediction$idx")
        }
        .reduce((a, b) => {
          a.join(b, "uniqueIdColumn")
        }).cache().showN_AndContinue(10)

    problemType match {
      case MultiClassClassificationProblem | BinaryClassificationProblem =>

        import org.apache.spark.sql.functions.{col, lit}

        import org.apache.spark.sql.functions.udf
        val predictionColumns: Array[Column] = dfWithPredictionsFromBaseModels.indices.toArray.map{ idx => col(s"prediction$idx")}

        val majorityFunc2 = udf { (s1: Double, s2: Double, s3: Double, s4: Double)  =>

          val occurances = mutable.Map.empty[Double, Int]
          Seq(s1,s2,s3,s4).foreach { x =>
            occurances.update(x, occurances.getOrElse(x, 0) + 1)
          }
          occurances.toArray.sortWith(_._2 > _._2).head._1
        }

        val majorityFunc = {
          val occurances = mutable.Map.empty[Column, Int]
          predictionColumns.foreach { x =>
            occurances.update(x, occurances.getOrElse(x, 0) + 1)
          }
          occurances.toArray.sortWith(_._2 > _._2).head._1
        }

        val mergedAndRegressedDF =
          joinedPredictions
            .showN_AndContinue(100)
            .withColumn("majority_prediction", majorityFunc2(predictionColumns: _*))
            .withColumnRenamed("majority_prediction", "prediction")
//            .drop(predictionColumns.map(_.toString): _*)
            .join(dfWithPredictionsFromBaseModels.head.drop("prediction"), Seq("uniqueIdColumn"), joinType = "left_outer")
            .cache()
            .showAllAndContinue

        results.foreach(_._2.dfWithPredictions.unpersist()) //TODO doubling see below

        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("indexedLabel")
          .setPredictionCol("prediction")
          .setMetricName("accuracy")

        val f1 = evaluator.evaluate(mergedAndRegressedDF)
        logger.info(s"$name : F1 = " + f1)
        FitnessResult(Map("f1" -> f1), problemType, mergedAndRegressedDF)

      case RegressionProblem =>

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

}

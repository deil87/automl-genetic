package com.automl.classifier.ensemble.bagging

import com.automl.{ConsistencyChecker, LogLossCustom, PaddedLogging}
import com.automl.dataset.{RandomSampling, StratifiedSampling}
import com.automl.evolution.dimension.hparameter.{BaggingHPGroup, HyperParametersField, HyperParametersGroup, MutableHParameter}
import com.automl.evolution.evaluation.EvaluationContextInfo
import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.regressor.{AverageRegressor, EnsemblingRegressor, MajorityVoteRegressor}
import com.automl.template.ensemble.bagging.BaggingMember
import com.automl.template.{TemplateMember, TemplateTree, TreeContext}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql._
import utils.{BenchmarkHelper, SparkMLUtils}
import org.apache.spark.ml.linalg.{DenseVector, Vector => MLVector}

import scala.collection.{immutable, mutable}

//TODO consider moving closer to BaggingMember
case class SparkGenericBagging(hpg: BaggingHPGroup = BaggingHPGroup())(implicit val logPaddingSize: Int = 0) extends BaggingMember
  with PaddedLogging with ConsistencyChecker{

  import utils.SparkMLUtils._

  var hpGroupInternal: HyperParametersGroup[_ <: MutableHParameter[Double, _]] = hpg

  override def ensemblingRegressor[_](problemType: ProblemType): EnsemblingRegressor[_] = problemType match {
    case RegressionProblem => new AverageRegressor()
    case MultiClassClassificationProblem | BinaryClassificationProblem => new MajorityVoteRegressor()
  }

  override def ensemblingFitnessError[A <: TemplateMember](trainDF: DataFrame,
                                                           testDF: DataFrame,
                                                           subMembers: Seq[TemplateTree[A]],
                                                           problemType: ProblemType,
                                                           hyperParamsMap: Option[HyperParametersField],
                                                           seed: Long)
                                                          (implicit tc: TreeContext): FitnessResult = {

    import trainDF.sparkSession.implicits._
    import org.apache.spark.sql.functions._

    val THE_LESS_THE_BETTER = ! theBiggerTheBetter(problemType)

    debug(s"Evaluating $name")
    debug(s"Sampling(stratified/random) without replacement for submembers of $name")
    val sampler = new RandomSampling
    val trainingSamplesForSubmembers = subMembers.zipWithIndex.map { case (model, idx) =>
//      val samplingSeed = new Random(seed).nextLong()//seed + idx
      val sample = sampler.sampleRatio(trainDF, 0.5, seed + idx)
      (model, sample)
    }

    BenchmarkHelper.time("SparkBagging consistency check") {
      consistencyCheck {
        val numberOfLevels = trainingSamplesForSubmembers.map(_._2.select("indexedLabel").distinct().count())
        if (!numberOfLevels.forall(_ == numberOfLevels.head)) {
          throw new IllegalStateException("Number of levels should be preserved during stratified sampling for submemers of Bagging ensemble node.")
        }
      }
    }

    val results: Seq[(TemplateTree[A], FitnessResult)] = trainingSamplesForSubmembers
      .zipWithIndex
      .map { case ((model, trainingSample), modelIdx) =>
        (model, model.evaluateFitness(trainingSample, testDF, problemType, hyperParamsMap))
      }

    val fitnessWeightColName = "fitness_weight"

    val dfWithPredictionsFromBaseModels: Seq[DataFrame] = results
      .map{r => r._2.dfWithPredictions.withColumn(fitnessWeightColName, lit(r._2.getCorrespondingMetric)).cache()} // TODO getCorrespondingMetric should be parametrized with metric

    import SparkMLUtils._

    val joinedPredictions: Dataset[Row] =
      dfWithPredictionsFromBaseModels
        .zipWithIndex
        .map{case (df, idx) =>
          df.select($"uniqueIdColumn", $"prediction" as s"prediction$idx", $"$fitnessWeightColName" as s"$fitnessWeightColName$idx", $"probability" as s"probability$idx")
        }
        .reduce((a, b) => {
          a.join(b, "uniqueIdColumn")
        }).cache()
//        .showN_AndContinue(10, "After joining predictions, weights and probabilities")

    def generateMajorityVoteUDF = {
      import org.apache.spark.sql.functions.udf
      udf { (preds: MLVector, weights: MLVector) =>
        val weightedOccurances: mutable.Map[Double, Double] = mutable.Map.empty
        preds.toArray.zipWithIndex.foreach { case (x, basePredictorIdx) =>
          // Calculating ranks of weights
          val rankCoef: Map[Int, Int] = weights.toArray.zipWithIndex.sortWith(_._1 < _._1).zipWithIndex
            .map{case ((weight, idx), rank) => (idx, rank + 1)}.toMap

          val adjustedRankCoef = rankCoef(basePredictorIdx).toDouble / 2
          val weightedAddition: Double = weights.apply(basePredictorIdx) * adjustedRankCoef
          val before: Double = weightedOccurances.getOrElse(x, 0)
          val newWeightedSum = before + weightedAddition
          weightedOccurances.update(x, newWeightedSum)
        }
        weightedOccurances.toArray.sortWith(_._2 > _._2).head._1  // rewrite with just finding argmax
      }
    }

    def generateProbabilityBasedPredictionUDF = {
      import org.apache.spark.sql.functions.udf
      udf { probabiliites: MLVector =>
        probabiliites.argmax.toDouble
      }
    }

    //TODO consider to remove as `generateProbabilitiesUDF` provides reasonable rawPrediction -> probabilities transformation
    /*def generateVotesUDF = {
      import org.apache.spark.sql.functions.udf
      udf { (preds: MLVector, weights: MLVector) =>
        val weightedOccurances: mutable.Map[Double, Double] = mutable.Map.empty
        new DenseVector(preds.toArray.zipWithIndex.map { case (x, basePredictorIdx) =>
          // Calculating ranks of weights
          val rankCoef: Map[Int, Int] = weights.toArray.zipWithIndex.sortWith(_._1 < _._1).zipWithIndex
            .map{case ((weight, idx), rank) => (idx, rank + 1)}.toMap

          val adjustedRankCoef = rankCoef(basePredictorIdx).toDouble / 2
          val weightedAddition: Double = weights.apply(basePredictorIdx) * adjustedRankCoef
          weightedAddition
//          val before: Double = weightedOccurances.getOrElse(x, 0)
//          val newWeightedSum = before + weightedAddition
//          weightedOccurances.update(x, newWeightedSum)
        })
      }
    }*/

    /**
      * In case base models agree on the predicted class => probability is 1.0
      * In case base models don't agree, we use weight to compute corresponding probabilities. Probabilities for all the other classes are set to 0.
      * @param preds predictions from base models. Has nothing to do with number of classes of the task. Zero based indexing.
      * @return
      */
    def generateWeightedProbabilitiesUDF = {
      import org.apache.spark.sql.functions.udf
      udf { ( preds: MLVector, weights: MLVector, baseProbabilities: MLVector) =>

        val predsAsArray = preds.toArray
        val numberOfPredictors = predsAsArray.length
        val numberOfClassesAsDouble = baseProbabilities.size.toDouble / numberOfPredictors
        val numberOfClasses = numberOfClassesAsDouble.toInt
        require(numberOfClasses == numberOfClassesAsDouble, "Loosing data when casting double to int")

        /**
          * Example: Map( class 1.0 -> Array( (again class, index of base predictor)),... )
          * As some of the classes may end up without votes for them,  we will get sparse `groupedPredictedClasses` map
          */
        val groupedPredictedClasses: Map[Double, Array[(Double, Int)]] = predsAsArray
          .zipWithIndex.groupBy(_._1)

        val paddedWithAllClasses = 0 until numberOfClasses map(classIdx => (classIdx, groupedPredictedClasses.getOrElse(classIdx, Array.empty[(Double, Int)])))
        val aggregatedSums: immutable.Seq[Double] = paddedWithAllClasses
          .map{ case (predictedClass, predictedClassToPredictorIndexArr) =>

            predictedClassToPredictorIndexArr.map{case (_, predictorIndex) =>
              // TODO OPTIMIZE: so we actually need base probabilities for predicted classes only
              val indexForBaseProbabilityForPredictedClass = predictorIndex * numberOfClasses + predictedClass.toInt

              val baseProbabilityForPredictor = baseProbabilities.apply(indexForBaseProbabilityForPredictedClass)

              val exaggerationCoef = 1 // TODO Consider to make it a bagging's HP, or metalearner property
              // Weight is a fitness of base predictor
              val wightOfCorrespondingPredictor =  weights.apply(predictorIndex) * exaggerationCoef
              val weightForPredictor = if(THE_LESS_THE_BETTER) 1.0 / wightOfCorrespondingPredictor else wightOfCorrespondingPredictor
              baseProbabilityForPredictor * weightForPredictor
            }
            .reduceOption((item1, item2) => item1 + item2)
            // For the cases when base predictors did not vote for some classes at all ( sparsity) we will use (0.0, 42.0) so that 0.0 / 42.0 become 0.0
            .getOrElse(0.0)

          }
        val weightedRawProbabilityForPredictedClass: Array[Double] =  aggregatedSums.toArray
//          .map(sumTermsPerPredictedClass => sumTermsPerPredictedClass._1 / sumTermsPerPredictedClass._2 ).toArray

        val sumOfRawProbabilities = weightedRawProbabilityForPredictedClass.sum
        val calibratedProbabilities = weightedRawProbabilityForPredictedClass.map(prob => prob * (1.0 / sumOfRawProbabilities))

        // If check correctness enabled
        val rounded = BigDecimal(calibratedProbabilities.sum).setScale(5, BigDecimal.RoundingMode.HALF_UP).toDouble
        require(rounded == 1, s"Sum of probabilities was not equal to 1 (${calibratedProbabilities.sum})")

        // TODO we might want to return rawPredictions ( rawProbabilities) as well.
        // Note: Calibrated probabilities mask original probabilities especially when predictors agree and normalisation/calibration makes probability equal to 1.0
        new DenseVector(calibratedProbabilities)

      }
    }

    def markDisputableInstanceUDF = {
      import org.apache.spark.sql.functions.udf
      udf { v: MLVector =>
        if(v.toArray.distinct.length > 1) 1 else 0
      }
    }

    problemType match {
      case MultiClassClassificationProblem | BinaryClassificationProblem =>

        BenchmarkHelper.time("checkThatWeHaveSameSetsOfCategoricalLevelsForAllSubmembers") {
          checkThatWeHaveSameSetsOfCategoricalLevelsForAllSubmembers(trainingSamplesForSubmembers)(trainDF.sparkSession)
        }

        import org.apache.spark.sql.functions.col
        val predictionColumns: Array[Column] = dfWithPredictionsFromBaseModels.indices.toArray.map{ idx => col(s"prediction$idx")}

        val baseModelsPredictionsColName = "baseModelsPredictionsColName"

        def predictionsAssembler = new VectorAssembler()
          .setInputCols(predictionColumns.map(_.toString))
          .setOutputCol(baseModelsPredictionsColName)

        val fitnessWeightColumns: Array[Column] = dfWithPredictionsFromBaseModels.indices.toArray.map{ idx => col(s"fitness_weight$idx")}

        val baseModelsFitnessWeights = "base_models_fitness_weights"

        def fitnessWeightsAssembler = new VectorAssembler()
          .setInputCols(fitnessWeightColumns.map(_.toString))
          .setOutputCol(baseModelsFitnessWeights)

        val probabilitiesColumns: Array[Column] = dfWithPredictionsFromBaseModels.indices.toArray.map{ idx => col(s"probability$idx")}

        // Combined probabilities from all base predictors. e.g.   [ prob_classA_pred1, prob_classB_pred1, prob_classA_pred2, prob_classB_pred2]
        val baseModelsProbabilities = "base_models_probabilities"
        def probabilitiesAssembler = new VectorAssembler()
          .setInputCols(probabilitiesColumns.map(_.toString))
          .setOutputCol(baseModelsProbabilities)

        val mergedAndRegressedDF =
            joinedPredictions
              .applyTransformation(predictionsAssembler)
              .applyTransformation(fitnessWeightsAssembler)
              .applyTransformation(probabilitiesAssembler)
              // TODO `majority_prediction` could be skipped as it is just experimental
              .withColumn("majority_prediction", generateMajorityVoteUDF($"$baseModelsPredictionsColName", $"$baseModelsFitnessWeights"))
              .withColumn("probability", generateWeightedProbabilitiesUDF( $"$baseModelsPredictionsColName", $"$baseModelsFitnessWeights", $"$baseModelsProbabilities"))
              .withColumn("prediction", generateProbabilityBasedPredictionUDF( $"probability"))
              .withColumn("isDisputable", markDisputableInstanceUDF($"$baseModelsPredictionsColName")) // TODO this is for debug purposes
              .join(dfWithPredictionsFromBaseModels.head.select("uniqueIdColumn", "indexedLabel"), Seq("uniqueIdColumn"), joinType = "left_outer")
              .withColumn("misclassified", $"prediction" =!= $"indexedLabel") // TODO this is for debug purposes
              .showN_AndContinue(1800, "With weighted raw probabilities (and unused majority predictions)")
              .cache()

//        mergedAndRegressedDF.show(21, false)

        results.foreach(_._2.dfWithPredictions.unpersist()) //TODO doubling see below

        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("indexedLabel")
          .setPredictionCol("prediction")
          .setMetricName("f1")

        val f1 = evaluator
          .evaluate(mergedAndRegressedDF)

        val logLoss = LogLossCustom.compute(mergedAndRegressedDF) // TODO we need a parameter that specifies metric so that we don't compute all of them every time

        debug(s"$name : f1 = " + f1 + ", logloss = " + logLoss)
        FitnessResult(Map("f1" -> f1, "logloss" -> logLoss), problemType, mergedAndRegressedDF)

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

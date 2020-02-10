package com.automl.evolution.evaluation

import com.automl.dataset.StratifiedRowsSampling
import com.automl.evolution.dimension.TemplateEvolutionDimension
import com.automl.evolution.dimension.hparameter._
import com.automl.population.HPPopulation
import com.automl.problemtype.ProblemType
import com.automl.template.simple.{Bayesian, DecisionTree, LogisticRegressionModel}
import com.automl.{ConfigProvider, PaddedLogging}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable

/**
  * It combines evaluation on base models and N best templates from template dimension.
  */
class HyperParameterMixedEvaluator(parentTemplateEvDimension: TemplateEvolutionDimension)(implicit val logPaddingSize: Int)
  extends PopulationEvaluator[HPPopulation, HyperParametersField, EvaluatedHyperParametersField] with PaddedLogging{

  override type CacheKeyType = ( HyperParametersField, Long)

  override def evaluateIndividuals(population: HPPopulation,
                                   workingDF: DataFrame,
                                   problemType: ProblemType,
                                   evaluationContextInfo: EvaluationContextInfo,
                                   seed: Long)
                                  (implicit cache: mutable.Map[( HyperParametersField, Long), Double]): Seq[EvaluatedHyperParametersField] = {
    val numberOfBestTemplates = 3
    val hpdConfig = ConfigProvider.config.getConfig("evolution.hyperParameterDimension")
    val samplingRatio = hpdConfig.getDouble("evaluationSamplingRatio")
    val sampledWorkingDF = new StratifiedRowsSampling().sampleRatio(workingDF, samplingRatio, seed).cache() //TODO every time we will compute and therefore deal with different damples.
    val sampledWorkingDFCount = sampledWorkingDF.count()
    debug(s"Sampling of the workingDF for hyper parameter evaluations ( $sampledWorkingDFCount out of ${workingDF.count()} )")
    // Note: there are multiple strategies of evaluating hps for template population.
    // 1) estimate base model/ building blocks of the templates(ensembles)
    // 2) estimate on last survived population(part of it)
    // 3) estimate on hallOfFame models (but we more conserned about population we are evolving at hands)
    // 4) ideally we need tu estimate parameters for all possible positions of the models in the ensembles.... this is for the future versions.
    // 5) mixture of 3) and 1)
//    BenchmarkHelper.time("Hyper-parameter evaluatePopulation ") {
      val threeBestTemplates = parentTemplateEvDimension.getEvaluatedPopulation.sortWith((a, b) => a.fitness.compareTo(b.fitness) > 0).map(_.template).take(numberOfBestTemplates)

    val currentLogPadding = logPaddingSize + 4

    threeBestTemplates.foreach(template => template.setLogPadding(currentLogPadding))

      val Array(trainingSplit, testSplit) = sampledWorkingDF.randomSplit(Array(0.67, 0.33), seed) // TODO move to Config ratio
      trainingSplit.cache()
      testSplit.cache()
      population.individuals.map { hpField =>
        val cacheKey = (hpField, sampledWorkingDFCount)
        val cacheKeyHashCode = cacheKey.hashCode()
        if (cache.isDefinedAt(cacheKey)) {
          debug(s"Cache hit happened for individual: $hpField")
          debug(s"Retrieved value from the cache with hashCode = $cacheKeyHashCode : ${cache(cacheKey)}")
        }
        val fitness = cache.getOrElseUpdate(cacheKey, {
          // Estimating 1) building blocks
          debug(s"Evaluating 1nd term for hpfield on base models:")
          val metricsFromBaseModels = hpField.modelsHParameterGroups.map {
            case hpGroup@BayesianHPGroup(_) =>
              val metric = Bayesian(hpGroup)(currentLogPadding).fitnessError(trainingSplit, testSplit, problemType).getCorrespondingMetric
              // We should get last Best Population from the TemplateCoevolution and estimate on the whole population or representative sample
              metric
            case hpGroup@LogisticRegressionHPGroup(_) =>
              val metric = LogisticRegressionModel(hpGroup)(currentLogPadding).fitnessError(trainingSplit, testSplit, problemType).getCorrespondingMetric
              metric
            case hpGroup@DecisionTreeHPGroup(_) =>
              val metric = DecisionTree(hpGroup)(currentLogPadding).fitnessError(trainingSplit, testSplit, problemType).getCorrespondingMetric
              metric
            case _ => throw new IllegalStateException("Unmatched HPGroup found in HP's evaluatePopulation method")
          }
          //TODO make sure that when our corresponding metric is "the less the better" we properly compare results
          // Estimating 2)
          debug(s"Evaluating 2nd term for hpfield  on ${threeBestTemplates.size} best templates in current template population:")
          val threeBestEvaluations = threeBestTemplates.map(template => template.evaluateFitness(trainingSplit, testSplit, problemType, hpFieldFromCoevolution = Some(hpField)).getCorrespondingMetric)
          val totalSumMetric = metricsFromBaseModels.sum + threeBestEvaluations.sum // we sum all metrics from each ModelHPGroup inn the field so that we can later decide which Field is the best
          debug(s"Entry $hpField with hashCode = ${cacheKey.hashCode()} was added to the cache with score = $totalSumMetric")
          totalSumMetric

        })
        EvaluatedHyperParametersField(hpField, fitness, problemType)
      }
//    }
  }
}

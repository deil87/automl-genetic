package com.automl.evolution.evaluation

import com.automl.dataset.StratifiedSampling
import com.automl.evolution.dimension.TemplateEvolutionDimension
import com.automl.evolution.dimension.hparameter._
import com.automl.population.HPPopulation
import com.automl.problemtype.ProblemType
import com.automl.template.simple.{Bayesian, DecisionTree, LogisticRegressionModel}
import com.automl.{ConfigProvider, PaddedLogging}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable

class HyperParameterContemporaryPopulationEvaluator(parentTemplateEvDimension: TemplateEvolutionDimension)(implicit val logPaddingSize: Int)
  extends PopulationEvaluator[HPPopulation, HyperParametersField, EvaluatedHyperParametersField] with PaddedLogging{

  override type CacheKeyType = ( HyperParametersField, Long)

  override def evaluateIndividuals(population: HPPopulation, workingDF: DataFrame, problemType: ProblemType, seed: Long)
                                  (implicit cache: mutable.Map[( HyperParametersField, Long), Double]): Seq[EvaluatedHyperParametersField] = {
    val numberOfBestTemplates = 3
    val hpdConfig = ConfigProvider.config.getConfig("evolution.hyperParameterDimension")
    val samplingRatio = hpdConfig.getDouble("evaluationSamplingRatio")
    val sampledWorkingDF = new StratifiedSampling().sample(workingDF, samplingRatio, seed).cache() //TODO every time we will compute and therefore deal with different samples.
    val sampledWorkingDFCount = sampledWorkingDF.count()
    debug(s"Sampling of the workingDF for hyper parameter evaluations ( $sampledWorkingDFCount out of ${workingDF.count()} )")

    val tPopulation = parentTemplateEvDimension.getPopulation

    val currentLogPadding = logPaddingSize + 4

    tPopulation.individuals.foreach(template => template.setLogPadding(currentLogPadding))

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

          //TODO make sure that when our corresponding metric is "the less the better" we properly compare results
          debug(s"Evaluating hpfield  on ${tPopulation.size}  templates in current template population:")
          val threeBestEvaluations = tPopulation.individuals.map(template => template.evaluateFitness(trainingSplit, testSplit, problemType, hyperParamsMap = Some(hpField)).getCorrespondingMetric)
          val totalSumMetric = threeBestEvaluations.sum
          debug(s"Entry $hpField with hashCode = ${cacheKey.hashCode()} was added to the cache with score = $totalSumMetric")
          totalSumMetric

        })
        EvaluatedHyperParametersField(hpField, fitness)
      }
  }
}

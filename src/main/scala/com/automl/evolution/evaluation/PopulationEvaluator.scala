package com.automl.evolution.evaluation

import com.automl.{Evaluated, EvaluatedTemplateData}
import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.helper.FitnessResult
import com.automl.population.Population
import com.automl.problemtype.ProblemType
import com.automl.template.{TemplateMember, TemplateTree}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable

trait PopulationEvaluator[PopulationType <: Population[IndividualType], IndividualType, EvaluatedResult<: Evaluated[EvaluatedResult]] {

  type CacheKeyType

  def evaluateIndividuals(population: PopulationType,
                          workingDF: DataFrame,
                          problemType: ProblemType,
                          evaluationContextInfo: EvaluationContextInfo,
                          seed: Long)
                         (implicit cache: mutable.Map[CacheKeyType, EvaluatedResult#FitnessType]): Seq[EvaluatedResult]
}
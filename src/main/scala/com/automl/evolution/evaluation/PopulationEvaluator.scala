package com.automl.evolution.evaluation

import com.automl.EvaluatedTemplateData
import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.template.{TemplateMember, TemplateTree}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable

trait PopulationEvaluator[PopulationType] {

  def evaluateIndividuals(population: PopulationType,
                          workingDataSet: DataFrame,
                          hyperParamField: HyperParametersField,
                          problemType: ProblemType)
                         (implicit cache: mutable.Map[(TemplateTree[TemplateMember], Long), FitnessResult]): Seq[EvaluatedTemplateData]
}
package com.automl.classifier.ensemble.bagging

import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.template.{TemplateMember, TemplateTree}
import org.apache.spark.sql.DataFrame

trait ReducingVarianceBasedOnDataStrategy {

  def generateTrainingSamples[A <: TemplateMember](trainDF: DataFrame,
                                                   subMembers: Seq[TemplateTree[A]],
                                                   hyperParamsMap: Option[HyperParametersField],
                                                   seed: Long): Seq[(TemplateTree[A], DataFrame)]
}

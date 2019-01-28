package com.automl.template.ensemble.cascading

import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.regressor.EnsemblingRegressor
import com.automl.template.{EvaluationMagnet, TemplateMember, TemplateTree, TreeContext}
import org.apache.spark.sql.DataFrame

case class MyCascadingImpl() extends CascadingMember {
  override def name: String = "MyCascadingImpl " + super.name

  override def ensemblingFitnessError[A <: TemplateMember](trainDF: DataFrame,
                                                           testDF: DataFrame,
                                                           subMembers: Seq[TemplateTree[A]],
                                                           problemType: ProblemType)
                                                          (implicit tc: TreeContext = TreeContext()): FitnessResult = ???

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

  override def ensemblingRegressor: EnsemblingRegressor = ???
}

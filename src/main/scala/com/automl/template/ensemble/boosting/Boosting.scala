package com.automl.template.ensemble.boosting

import com.automl.helper.FitnessResult
import com.automl.regressor.EnsemblingRegressor
import com.automl.template.{EvaluationMagnet, TemplateMember, TemplateTree, TreeContext}
import org.apache.spark.sql.DataFrame

case class Boosting() extends BoostingMember {
  override def name: String = "Boosting " + super.name


  override def ensemblingFitnessError[A <: TemplateMember](trainDF: DataFrame,
                                                           testDF: DataFrame,
                                                           subMembers: Seq[TemplateTree[A]])
                                                          (implicit tc: TreeContext = TreeContext()): FitnessResult = ???

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

  override def fitnessError(trainDF: DataFrame, testDF: DataFrame): FitnessResult = ???


  override def ensemblingRegressor: EnsemblingRegressor = ???
}


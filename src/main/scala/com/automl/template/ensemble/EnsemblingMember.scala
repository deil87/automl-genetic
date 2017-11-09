package com.automl.template.ensemble

import com.automl.helper.FitnessResult
import com.automl.regressor.{AverageRegressor, EnsemblingRegressor}
import com.automl.template._
import com.automl.template.ensemble.bagging.BaggingMember
import com.automl.template.ensemble.boosting.BoostingMember
import com.automl.template.ensemble.cascading.CascadingMember
import com.automl.template.ensemble.stacking.StackingMember
import org.apache.spark.sql._

trait EnsemblingMember extends TemplateMember {
  override def name: String = "ensembling member"

  def ensemblingFitnessError[A <: TemplateMember](trainDF: DataFrame,
                                                  testDF: DataFrame,
                                                  subMembers: Seq[TemplateTree[A]])
                                                 (implicit tc: TreeContext = TreeContext()): FitnessResult

  def ensemblingRegressor: EnsemblingRegressor
}


object EnsemblingMember {
  val poolOfEnsemblingModels =
    BaggingMember.poolOfBaggingModels ++ BoostingMember.poolOfBoostingModels + StackingMember.MyStackingImpl + CascadingMember.MyCascadingImpl
}

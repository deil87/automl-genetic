package com.automl.template.ensemble

import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.regressor.{AverageRegressor, EnsemblingRegressor}
import com.automl.template._
import com.automl.template.ensemble.bagging.BaggingMember
import com.automl.template.ensemble.boosting.BoostingMember
import com.automl.template.ensemble.cascading.CascadingMember
import com.automl.template.ensemble.stacking.StackingMember
import org.apache.spark.sql._

trait EnsemblingModelMember extends TemplateMember {
  override def name: String = "ensembling member"

  //TODO maybe we can reuse the same name so that we can treat SImple and Ensempling nodes equally?
  override def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType): FitnessResult = {
    throw new IllegalStateException("We should call ensemblingFitnessError method for ensembling classifiers")
  }

  def ensemblingFitnessError[A <: TemplateMember](trainDF: DataFrame,
                                                  testDF: DataFrame,
                                                  subMembers: Seq[TemplateTree[A]],
                                                  problemType: ProblemType,
                                                  hyperParamsMap: HyperParametersField)
                                                 (implicit tc: TreeContext = TreeContext()): FitnessResult

}


object EnsemblingModelMember {
  val poolOfEnsemblingModels: Set[EnsemblingModelMember] =
    BaggingMember.poolOfBaggingModels + StackingMember.MyStackingImpl /*++ BoostingMember.poolOfBoostingModels + StackingMember.MyStackingImpl + CascadingMember.MyCascadingImpl*/

  def randomMember: EnsemblingModelMember = poolOfEnsemblingModels.toSeq.randElement
}

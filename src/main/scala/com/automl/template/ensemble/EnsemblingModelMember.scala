package com.automl.template.ensemble

import com.automl.{ConfigProvider, PaddedLogging}
import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.template._
import com.automl.template.ensemble.bagging.BaggingMember
import com.automl.template.ensemble.stacking.{GenericStacking, StackingMember}
import org.apache.spark.sql._

import scala.collection.JavaConverters._

trait EnsemblingModelMember extends TemplateMember { self: PaddedLogging =>
  override def name: String = "ensembling member"

  //TODO maybe we can reuse the same name so that we can treat Simple and Ensempling nodes equally?
  override def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType, hyperParametersField: Option[HyperParametersField]): FitnessResult = {
    throw new IllegalStateException("We should call ensemblingFitnessError method for ensembling classifiers")
  }

  def ensemblingFitnessError[A <: TemplateMember](trainDF: DataFrame,
                                                  testDF: DataFrame,
                                                  subMembers: Seq[TemplateTree[A]],
                                                  problemType: ProblemType,
                                                  hyperParamsMap: Option[HyperParametersField],
                                                  seed: Long)
                                                 (implicit tc: TreeContext = TreeContext()): FitnessResult

}


object EnsemblingModelMember {
  val poolOfEnsemblingModels: Set[EnsemblingModelMember] =
    BaggingMember.poolOfBaggingModels + StackingMember.MyStackingImpl /*++ BoostingMember.poolOfBoostingModels + StackingMember.MyStackingImpl + CascadingMember.MyCascadingImpl*/

  val tdConfig = ConfigProvider.config.getConfig("evolution.templateDimension")

  lazy val poolOfEnsemblingModelsNames: Seq[String] = tdConfig.getStringList("poolOfEnsemblingModels").asScala

  def poolOfEnsemblingModelsByNames(names: Seq[String]): Seq[EnsemblingModelMember] = names.flatMap {
    case "stacking" => BaggingMember.poolOfBaggingModels
    case "bagging" => Set[EnsemblingModelMember](StackingMember.MyStackingImpl)
  }

  def randomMember: EnsemblingModelMember = poolOfEnsemblingModelsByNames(poolOfEnsemblingModelsNames).randElement
}

package com.automl.template.ensemble.boosting

import com.automl.PaddedLogging
import com.automl.evolution.dimension.hparameter.{HyperParametersField, HyperParametersGroup, MutableHParameter}
import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.regressor.EnsemblingRegressor
import com.automl.template.{EvaluationMagnet, TemplateMember, TemplateTree, TreeContext}
import org.apache.spark.sql.DataFrame

case class Boosting()(implicit val logPaddingSize: Int = 0) extends BoostingMember with PaddedLogging{
  override def name: String = "Boosting " + super.name

  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = {
    case BinaryClassificationProblem => ??? // TODO enable
    case MultiClassClassificationProblem => ???
    case RegressionProblem => ??? // TODO enable
  }

  override def ensemblingFitnessError[A <: TemplateMember](trainDF: DataFrame,
                                                           testDF: DataFrame,
                                                           subMembers: Seq[TemplateTree[A]],
                                                           problemType: ProblemType,
                                                           hyperParamsMap: Option[HyperParametersField],
                                                           seed: Long)
                                                          (implicit tc: TreeContext = TreeContext()): FitnessResult = ???

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

  //TODO move to constructor
  var hpGroupInternal: HyperParametersGroup[_ <: MutableHParameter[Double, _]] = ???
}


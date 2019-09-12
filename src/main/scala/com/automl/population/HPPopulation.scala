package com.automl.population

import com.automl.evolution.dimension.hparameter._
import com.automl.template.{TemplateMember, TemplateTree}


class HPPopulation(val individuals: Seq[ HyperParametersField])
  extends Population[HyperParametersField] {
  override def render: Unit = ???
}

object HPPopulation {
  def randomPopulation = new HPPopulation(
    Seq(
      HyperParametersField(
        Seq(
          BayesianHPGroup(), LogisticRegressionHPGroup(), DecisionTreeHPGroup(), RandomForestHPGroup()
        )
      )
    )
  )

  private def allPossibleHPGroups = Set(BayesianHPGroup(), LogisticRegressionHPGroup(), DecisionTreeHPGroup(), RandomForestHPGroup())

  def randomRelevantHPFieldFor(templateMember: TemplateMember) =
    HyperParametersField(
      allPossibleHPGroups.filter(_.isRelevantTo(templateMember)).toSeq
    )
}

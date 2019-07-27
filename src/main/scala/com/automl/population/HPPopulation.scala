package com.automl.population

import com.automl.evolution.dimension.hparameter.{BayesianHPGroup, DecisionTreeHPGroup, HyperParametersField, LogisticRegressionHPGroup}
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
          BayesianHPGroup(), LogisticRegressionHPGroup(), DecisionTreeHPGroup()
        )
      )
    )
  )

  private val allPossibleHPGroups = Set(BayesianHPGroup(), LogisticRegressionHPGroup(), DecisionTreeHPGroup())

  def randomRelevantHPFieldFor(templateMember: TemplateMember) =
    HyperParametersField(
      allPossibleHPGroups.filter(_.isRelevantTo(templateMember)).toSeq
    )
}

package com.automl.population

import com.automl.evolution.dimension.hparameter.{BayesianHPGroup, DecisionTreeHPGroup, HyperParametersField, LogisticRegressionHPGroup}


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

  def randomHPField =
    HyperParametersField(
      Seq(
        BayesianHPGroup(), LogisticRegressionHPGroup(), DecisionTreeHPGroup()
      )
    )
}

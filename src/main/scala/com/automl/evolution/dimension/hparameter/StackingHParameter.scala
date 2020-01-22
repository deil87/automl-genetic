package com.automl.evolution.dimension.hparameter

import com.automl.template.TemplateMember
import com.automl.template.ensemble.stacking.GenericStacking


case class StackingHPGroup(hpParameters:Seq[StackingHParameter] = Seq())
  extends HyperParametersGroup[StackingHParameter] {

  override def isRelevantTo(templateTree: TemplateMember): Boolean = templateTree.isInstanceOf[GenericStacking]

  override def mutate(): HyperParametersGroup[StackingHParameter] = {
    StackingHPGroup(hpParameters = hpParameters.map(hpModelTpe => hpModelTpe.mutate()))
  }
}

object StackingHPGroup {
  val default = LogisticRegressionHPGroup()
}

trait StackingHParameter extends MutableHParameter[Double, StackingHParameter]


package com.automl.evolution.dimension.hparameter

import com.automl.template.simple.LogisticRegressionModel
import com.automl.template.{TemplateMember, TemplateTree}

import scala.util.Random


case class BaggingHPGroup(hpParameters:Seq[StackingHParameter] = Seq())
  extends HyperParametersGroup[StackingHParameter] {

  override def isRelevantTo(templateTree: TemplateMember): Boolean = templateTree.isInstanceOf[LogisticRegressionModel]

  override def mutate(): HyperParametersGroup[StackingHParameter] = {
    StackingHPGroup(hpParameters = hpParameters.map(hpModelTpe => hpModelTpe.mutate()))
  }
}

object BaggingHPGroup {
  val default = BaggingHPGroup()
}

trait BaggingHParameter extends MutableHParameter[Double, StackingHParameter]


case class BaggingExaggeration(initialValue: Option[Double] = None) extends StackingHParameter with DoubleHPRange[StackingHParameter] { // we can specialize with Marker trait which parameter can be used with which Model
  override def min: Double = 0.0

  override def max: Double = 0.5

  override def step: Double = 0.1 //TODO can we change step during evolution? we need to detect stagnations/convergence and them change step for fine tuning.

  override def getDefaultRandomly: Double = round(new Random().nextDouble(), 1) // In theory we are interested not only on round values but on the best ones

  override def newInstance: StackingHParameter = BaggingExaggeration()

  override def toString: String = "lambda:" + currentValue.toString
}

package com.automl.evolution.dimension.hparameter

import com.automl.template.simple.LogisticRegressionModel
import com.automl.template.{TemplateMember, TemplateTree}

import scala.util.Random


case class BaggingHPGroup(hpParameters:Seq[BaggingHParameter] = Seq())
  extends HyperParametersGroup[BaggingHParameter] {

  override def isRelevantTo(templateTree: TemplateMember): Boolean = templateTree.isInstanceOf[LogisticRegressionModel]

  override def mutate(): HyperParametersGroup[BaggingHParameter] = {
    BaggingHPGroup(hpParameters = hpParameters.map(hpModelTpe => hpModelTpe.mutate()))
  }
}

object BaggingHPGroup {
  val default = LogisticRegressionHPGroup()
}

trait BaggingHParameter extends MutableHParameter[Double, BaggingHParameter]


case class BaggingExaggeration(initialValue: Option[Double] = None) extends BaggingHParameter with DoubleHPRange[BaggingHParameter] { // we can specialize with Marker trait which parameter can be used with which Model
  override def min: Double = 0.0

  override def max: Double = 0.5

  override def step: Double = 0.1 //TODO can we change step during evolution? we need to detect stagnations/convergence and them change step for fine tuning.

  override def getDefault: Double = round(new Random().nextDouble(), 1) // In theory we are interested not only on round values but on the best ones

  override def newInstance: BaggingHParameter = BaggingExaggeration()

  override def toString: String = "lambda:" + currentValue.toString
}

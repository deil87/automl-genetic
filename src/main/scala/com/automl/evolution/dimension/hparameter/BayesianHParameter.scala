package com.automl.evolution.dimension.hparameter

import com.automl.template.TemplateMember
import com.automl.template.simple.Bayesian


case class BayesianHPGroup(hpParameters:Seq[BayesianHParameter[Double]] = Seq(Smoothing()))
  extends HyperParametersGroup[BayesianHParameter[Double]] {

  override def isRelevantTo(templateTree: TemplateMember): Boolean = templateTree.isInstanceOf[Bayesian]

  // TODO consider using HList
  override def mutate(): HyperParametersGroup[BayesianHParameter[Double]] = {
    BayesianHPGroup(hpParameters = hpParameters.map(hpModelTpe => hpModelTpe.mutate()))
  }
}
object BayesianHPGroup {
  val default = BayesianHPGroup()
}

trait BayesianHParameter[T <: AnyVal] extends MutableHParameter[T, BayesianHParameter[T]]

case class Smoothing() extends BayesianHParameter[Double] with DoubleHPRange { // we can specialize with Marker trait which parameter can be used with which Model
  override def min: Double = 1.0

  override def max: Double = 10.0

  override def step: Double = 1.0

  override def getDefault: Double = getNextWithinTheRange

  var currentValue: Double = {
    val defaultValue = getDefault
//    explored(defaultValue) = true
    defaultValue
  }

  // Note: In DepthDependentTemplateMutationStrategy we use strategy to mutate.
  // Here it is opposite... class has mutate method in its API and we can pass Strategy as parameter.
  override def mutate(/*hpMutationStrategy: Strategy*/): Smoothing = {
    var newValue = getNextClosestWithinTheRange(currentValue)
    while(newValue == currentValue) {
//    while(explored.contains(newValue)) {
      newValue = getNextClosestWithinTheRange(currentValue)
    }
    val newVersion = Smoothing()
    newVersion.currentValue = newValue
//    explored(newVersion.currentValue) = true
    newVersion
  }


  override def toString: String = "smoothing:" + currentValue.toString
}
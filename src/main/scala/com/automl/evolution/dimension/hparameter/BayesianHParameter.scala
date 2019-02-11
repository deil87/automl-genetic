package com.automl.evolution.dimension.hparameter

import scala.util.Random

case class BayesianHPGroup(hpParameters:Seq[BayesianHParameter[Double]] = Seq(Smoothing()))
  extends HyperParametersGroup[BayesianHParameter[Double]] {

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

  var currentValue: Double = getDefault

  override def mutate(): Smoothing = {
    var newValue = getNextWithinTheRange
    while(newValue == currentValue) {
      newValue = getNextWithinTheRange
    }
    val newVersion = Smoothing()
    newVersion.currentValue = newValue
    newVersion
  }


  override def toString: String = "smoothing:" + currentValue.toString
}
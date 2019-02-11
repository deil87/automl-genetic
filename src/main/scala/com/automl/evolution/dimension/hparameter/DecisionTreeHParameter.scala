package com.automl.evolution.dimension.hparameter

import scala.math.BigDecimal.RoundingMode
import scala.util.Random


case class DecisionTreeHPGroup(hpParameters:Seq[DecisionTreeHParameter[Double]] = Seq(MaxDepth()))
  extends HyperParametersGroup[DecisionTreeHParameter[Double]] {

  override def mutate(): HyperParametersGroup[DecisionTreeHParameter[Double]] = {
    DecisionTreeHPGroup(hpParameters = hpParameters.map(hpModelTpe => hpModelTpe.mutate()))
  }
}

object DecisionTreeHPGroup {
  val default = DecisionTreeHPGroup()
}

trait DecisionTreeHParameter[T <: AnyVal] extends MutableHParameter[T, DecisionTreeHParameter[T]]

case class MaxDepth(initialValue: Option[Double] = None) extends DecisionTreeHParameter[Double] with DoubleHPRange { // we can specialize with Marker trait which parameter can be used with which Model
  override def min: Double = 1.0

  override def max: Double = 10.0

  override def step: Double = 1.0 //TODO can we change step during evolution? we need to detect stagnations/convergence and them change step for fine tuning.

  override def getDefault: Double = getNextWithinTheRange

  var currentValue: Double = initialValue.getOrElse(getDefault)

  override def mutate(): MaxDepth = {
    var newValue = getNextWithinTheRange
    while(newValue == currentValue) {
      newValue = getNextWithinTheRange
    }
    val newVersion = MaxDepth()
    newVersion.currentValue = newValue
    newVersion
  }

  override def toString: String = "max_depth:" + currentValue.toString
}
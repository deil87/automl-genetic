package com.automl.evolution.dimension.hparameter

import com.automl.template.TemplateMember
import com.automl.template.simple.DecisionTree


case class DecisionTreeHPGroup(hpParameters:Seq[DecisionTreeHParameter[Double]] = Seq(MaxDepth()))
  extends HyperParametersGroup[DecisionTreeHParameter[Double]] {

  override def isRelevantTo(templateTree: TemplateMember): Boolean = templateTree.isInstanceOf[DecisionTree]

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
    var newValue = getNextClosestWithinTheRange(currentValue)
    while(newValue == currentValue) {
      newValue = getNextClosestWithinTheRange(currentValue)
    }
    val newVersion = MaxDepth()
    newVersion.currentValue = newValue
    newVersion
  }

  override def toString: String = "max_depth:" + currentValue.toString
}
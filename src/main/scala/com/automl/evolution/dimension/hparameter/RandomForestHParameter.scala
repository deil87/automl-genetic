package com.automl.evolution.dimension.hparameter

import com.automl.template.TemplateMember
import com.automl.template.simple.RandomForest


case class RandomForestHPGroup(hpParameters:Seq[RandomForestHParameter[Double]] = Seq(MaxDepthRF(), NumTreesRF())) // TODO subsamplingRate, featureSubsetStrategy
  extends HyperParametersGroup[RandomForestHParameter[Double]] {

  override def isRelevantTo(templateTree: TemplateMember): Boolean = templateTree.isInstanceOf[RandomForest]

  override def mutate(): HyperParametersGroup[RandomForestHParameter[Double]] = {
    RandomForestHPGroup(hpParameters = hpParameters.map(hpModelTpe => hpModelTpe.mutate()))
  }
}

object RandomForestHPGroup {
  val default = RandomForestHPGroup()
}

trait RandomForestHParameter[T <: AnyVal] extends MutableHParameter[T, RandomForestHParameter[T]]


//TODO maybe we can reuse similar HPS. Need to think about how to preserve strong typing.
case class MaxDepthRF(initialValue: Option[Double] = None) extends RandomForestHParameter[Double] with DoubleHPRange[RandomForestHParameter[Double]] { // we can specialize with Marker trait which parameter can be used with which Model
  override def min: Double = 2.0

  override def max: Double = 7.0

  override def step: Double = 1.0 //TODO can we change step during evolution? we need to detect stagnations/convergence and them change step for fine tuning.

  override def getDefaultRandomly: Double = getNextWithinTheRange

  override def newInstance: RandomForestHParameter[Double] = MaxDepthRF()

  override def toString: String = "max_depth_rf:" + currentValue.toString
}

case class NumTreesRF(initialValue: Option[Double] = None) extends RandomForestHParameter[Double] with DoubleHPRange[RandomForestHParameter[Double]] {
  override def min: Double = 20.0

  override def max: Double = 20.0

  override def step: Double = 10.0

  override def getDefaultRandomly: Double = getNextWithinTheRange

  override def newInstance: RandomForestHParameter[Double] = NumTreesRF()

  override def toString: String = "num_trees_rf:" + currentValue.toString
}
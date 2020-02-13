package com.automl.evolution.dimension.hparameter

import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.template.simple.LogisticRegressionModel
import com.automl.template.{TemplateMember, TemplateTree}

import scala.util.Random


case class BaggingHPGroup(hpParameters:Seq[BaggingHParameter] =
                          Seq(BaggingReplacement(), BaggingExaggeration()/* BaggingRowsSamplingStrategy()*/, BaggingRowsSamplingRatio()/*, BaggingColumnsSamplingRatio()*/))
  extends HyperParametersGroup[BaggingHParameter] {

  override def isRelevantTo(templateTree: TemplateMember): Boolean = templateTree.isInstanceOf[SparkGenericBagging]

  override def mutate(): HyperParametersGroup[BaggingHParameter] = {
    BaggingHPGroup(hpParameters = hpParameters.map(hpModelTpe => hpModelTpe.mutate()))
  }
}

object BaggingHPGroup {
  val default = BaggingHPGroup()
}

trait BaggingHParameter extends MutableHParameter[Double, BaggingHParameter]


case class BaggingExaggeration(initialValue: Option[Double] = None) extends BaggingHParameter with DoubleHPRange[BaggingHParameter] { // we can specialize with Marker trait which parameter can be used with which Model
  override def min: Double = 0.0

  override def max: Double = 0.5

  override def step: Double = 0.1 //TODO can we change step during evolution? we need to detect stagnations/convergence and them change step for fine tuning.

  override def getDefaultRandomly: Double = getNextWithinTheRange // In theory we are interested not only on round values but on the best ones

  override def newInstance: BaggingHParameter = BaggingExaggeration()

  override def toString: String = "bagging exaggeration:" + currentValue.toString
}

// 1.0 stands for true, 0.0 stands for false
case class BaggingReplacement(initialValue: Option[Double] = None) extends BaggingHParameter with DoubleHPRange[BaggingHParameter] {
  override def min: Double = 0.0

  override def max: Double = 1.0

  override def step: Double = 1
  override def getDefaultRandomly: Double = getNextWithinTheRange // In theory we are interested not only on round values but on the best ones

  override def newInstance: BaggingHParameter = BaggingReplacement()

  override def toString: String = "bagging replacement:" + currentValue.toString
}

// 1.0 stands for Stratification, 0.0 stands for Random //TODO Disabled!
case class BaggingRowsSamplingStrategy(initialValue: Option[Double] = None) extends BaggingHParameter with DoubleHPRange[BaggingHParameter] {
  override def min: Double = 0.0

  override def max: Double = 1.0

  override def step: Double = 1
  override def getDefaultRandomly: Double = getNextWithinTheRange

  override def newInstance: BaggingHParameter = BaggingRowsSamplingStrategy()

  override def toString: String = "bagging rows sampling strategy:" + currentValue.toString
}

case class BaggingRowsSamplingRatio(initialValue: Option[Double] = None) extends BaggingHParameter with DoubleHPRange[BaggingHParameter] {
  override def min: Double = 0.5

  override def max: Double = 1.0

  override def step: Double = 0.1
  override def getDefaultRandomly: Double = getNextWithinTheRange

  override def newInstance: BaggingHParameter = BaggingRowsSamplingRatio()

  override def toString: String = "bagging rows sampling ratio:" + currentValue.toString
}

case class BaggingColumnsSamplingRatio(initialValue: Option[Double] = None) extends BaggingHParameter with DoubleHPRange[BaggingHParameter] {
  override def min: Double = 1.0 // disabling

  override def max: Double = 1.0

  override def step: Double = 0.1
  override def getDefaultRandomly: Double = getNextWithinTheRange

  override def newInstance: BaggingHParameter = BaggingColumnsSamplingRatio()

  override def toString: String = "bagging columns sampling ratio:" + currentValue.toString
}

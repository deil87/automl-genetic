package com.automl.evolution.dimension.hparameter

import com.automl.template.TemplateMember
import com.automl.template.simple.{RandomForest, SVMModel}

import scala.util.Random


case class SVMHPGroup(hpParameters:Seq[SVMHParameter[Double]] = Seq(MaxIterSVM(), RegParamSVM())) //TODO do we need Seq here?
  extends HyperParametersGroup[SVMHParameter[Double]] {

  override def isRelevantTo(templateTree: TemplateMember): Boolean = templateTree.isInstanceOf[SVMModel]

  override def mutate(): HyperParametersGroup[SVMHParameter[Double]] = {
    SVMHPGroup(hpParameters = hpParameters.map(hpModelTpe => hpModelTpe.mutate()))
  }
}

object SVMHPGroup {
  val default = SVMHPGroup()
}

trait SVMHParameter[T <: AnyVal] extends MutableHParameter[T, SVMHParameter[T]]


//TODO maybe we can reuse similar HPS. Need to think about how to preserve strong typing.
case class MaxIterSVM(initialValue: Option[Double] = None) extends SVMHParameter[Double] with DoubleHPRange[SVMHParameter[Double]] { // we can specialize with Marker trait which parameter can be used with which Model
  override def min: Double = 50.0

  override def max: Double = 200.0

  override def step: Double = 50.0 //TODO can we change step during evolution? we need to detect stagnations/convergence and them change step for fine tuning.

  override def getDefaultRandomly: Double = getNextWithinTheRange

  override def newInstance: SVMHParameter[Double] = MaxIterSVM()

  override def toString: String = "max_iter_svm:" + currentValue.toString
}

case class RegParamSVM(initialValue: Option[Double] = None) extends SVMHParameter[Double] with DoubleHPRange[SVMHParameter[Double]] { // we can specialize with Marker trait which parameter can be used with which Model
  override def min: Double = 0.0

  override def max: Double = 1.0

  override def step: Double = 0.1 //TODO can we change step during evolution? we need to detect stagnations/convergence and them change step for fine tuning.

  override def getDefaultRandomly: Double = round(new Random().nextDouble(), 1) // In theory we are interested not only on round values but on the best ones

  override def newInstance: SVMHParameter[Double] = RegParamSVM()

  override def toString: String = "reg_param_svm:" + currentValue.toString
}
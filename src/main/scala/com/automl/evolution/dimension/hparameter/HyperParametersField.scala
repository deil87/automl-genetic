package com.automl.evolution.dimension.hparameter


//case class HyperParametersField[T  <: MutableHParameter[Double, T]](modelsHParameterGroups: Seq[HyperParametersGroup[ T]]) {
//case class HyperParametersField(modelsHParameterGroups: Seq[HyperParametersGroup[MutableHParameter[Double, _]]]) {
//case class HyperParametersField(modelsHParameterGroups: Seq[HyperParametersGroup[ExT] forSome { type Ext <: MutableHParameter[Double, _]}] ) {
// Existential Types - I love you guys!
case class HyperParametersField(modelsHParameterGroups: Seq[HyperParametersGroup[_ <: MutableHParameter[Double, _]]] ) {

  override def hashCode(): Int = {
    modelsHParameterGroups.map(_.hpParameters.map(_.currentValue.hashCode()).sum).sum
  }

  def getLogisticRegressionHPGroup: HyperParametersGroup[_ <: MutableHParameter[Double, _]] = {
    modelsHParameterGroups.find{
      case LogisticRegressionHPGroup(_) => true
    }.get
  }

  override def toString: String = modelsHParameterGroups.map(group => group.hpParameters.map(parameter => s"$parameter").mkString(" , ")).mkString(" | ")
}

object HyperParametersField {
  def default = HyperParametersField(Seq(BayesianHPGroup(), LogisticRegressionHPGroup()))
}
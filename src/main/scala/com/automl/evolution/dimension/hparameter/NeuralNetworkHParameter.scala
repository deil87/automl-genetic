package com.automl.evolution.dimension.hparameter

import com.automl.template.TemplateMember
import com.automl.template.simple.NeuralNetwork


case class NeuralNetworkHPGroup(hpParameters:Seq[NeuralNetworkHParameter[Double]] =
                                Seq(HiddenLayers(), Neurons1stLayer(), Neurons2ndLayer(), Neurons3rdLayer(), Tolerance()))
  extends HyperParametersGroup[NeuralNetworkHParameter[Double]] {

  override def isRelevantTo(templateTree: TemplateMember): Boolean = templateTree.isInstanceOf[NeuralNetwork]

  override def mutate(): HyperParametersGroup[NeuralNetworkHParameter[Double]] = {
    NeuralNetworkHPGroup(hpParameters = hpParameters.map(hpModelTpe => hpModelTpe.mutate()))
  }
}

object NeuralNetworkHPGroup {
  val default = NeuralNetworkHPGroup()
}

trait NeuralNetworkHParameter[T <: AnyVal] extends MutableHParameter[T, NeuralNetworkHParameter[T]]

/*case class HiddenLayers(initialValue: Option[Seq[Double]] = None) extends NeuralNetworkHParameter[Seq[Double]] { // we can specialize with Marker trait which parameter can be used with which Model

  val maxNumberOfLayers = 3
  val maxNumberOfNeurons = 10

  override def getDefaultRandomly: Seq[Double] = {
    val rg = new Random()
    val numberOfLayers = rg.nextInt(3) + 1
    require(numberOfLayers <= maxNumberOfLayers)
    val res = (1 to numberOfLayers).map(layerIdx => {
      val numberOfNeurons = 2 + rg.nextInt(9).toDouble
      require(numberOfNeurons <= maxNumberOfNeurons)
      numberOfNeurons
    }) // better needs to be dependant on dataset
    res
  }


  override var currentValue: Seq[Double] = {
    val defaultValue = initialValue.getOrElse(getDefaultRandomly)
    defaultValue
  }

  override def mutate(): NeuralNetworkHParameter[Seq[Double]] = {
    val idxToMutate = new Random().nextInt(currentValue.size)
    val oldValue: Double = currentValue(idxToMutate)
    var newValue: Double = oldValue
    while( oldValue == newValue) {
      newValue = new Random().nextInt(11).toDouble
    }
    require(newValue <= maxNumberOfNeurons)
    currentValue.updated(idxToMutate, newValue)
    this
  }

  def newInstance: NeuralNetworkHParameter[Seq[Double]] = HiddenLayers()

  override def toString: String = "hidden_layers:" + currentValue.toString
}*/

case class HiddenLayers(initialValue: Option[Double] = None) extends NeuralNetworkHParameter[Double] with DoubleHPRange[NeuralNetworkHParameter[Double]] { // we can specialize with Marker trait which parameter can be used with which Model
  override def min: Double = 1.0

  override def max: Double = 2.0

  override def step: Double = 1.0

  override def getDefaultRandomly: Double = getNextWithinTheRange

  override def newInstance: NeuralNetworkHParameter[Double] = HiddenLayers()

  override def toString: String = "hidden_layers:" + currentValue.toString
}

case class Neurons1stLayer(initialValue: Option[Double] = None) extends NeuralNetworkHParameter[Double] with DoubleHPRange[NeuralNetworkHParameter[Double]] { // we can specialize with Marker trait which parameter can be used with which Model
  override def min: Double = 2.0

  override def max: Double = 10.0

  override def step: Double = 1.0

  override def getDefaultRandomly: Double = getNextWithinTheRange

  override def newInstance: NeuralNetworkHParameter[Double] = Neurons1stLayer()

  override def toString: String = "neurons_1st_layer:" + currentValue.toString
}

case class Neurons2ndLayer(initialValue: Option[Double] = None) extends NeuralNetworkHParameter[Double] with DoubleHPRange[NeuralNetworkHParameter[Double]] { // we can specialize with Marker trait which parameter can be used with which Model
  override def min: Double = 2.0

  override def max: Double = 10.0

  override def step: Double = 1.0

  override def getDefaultRandomly: Double = getNextWithinTheRange

  override def newInstance: NeuralNetworkHParameter[Double] = Neurons2ndLayer()

  override def toString: String = "neurons_2nd_layer:" + currentValue.toString
}

case class Neurons3rdLayer(initialValue: Option[Double] = None) extends NeuralNetworkHParameter[Double] with DoubleHPRange[NeuralNetworkHParameter[Double]] { // we can specialize with Marker trait which parameter can be used with which Model
  override def min: Double = 2.0

  override def max: Double = 10.0

  override def step: Double = 1.0

  override def getDefaultRandomly: Double = getNextWithinTheRange

  override def newInstance: NeuralNetworkHParameter[Double] = Neurons3rdLayer()

  override def toString: String = "neurons_3rd_layer:" + currentValue.toString
}

case class Tolerance(initialValue: Option[Double] = None) extends NeuralNetworkHParameter[Double] with DoubleHPRange[NeuralNetworkHParameter[Double]] { // we can specialize with Marker trait which parameter can be used with which Model
  override def min: Double = 2.0

  override def max: Double = 7.0

  override def step: Double = 1.0 //TODO can we change step during evolution? we need to detect stagnations/convergence and them change step for fine tuning.

  override def getDefaultRandomly: Double = getNextWithinTheRange

  override def newInstance: NeuralNetworkHParameter[Double] = Tolerance()

  override def toString: String = "max_depth:" + currentValue.toString
}
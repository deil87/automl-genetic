package com.automl.evolution.dimension.hparameter

import scala.math.BigDecimal.RoundingMode
import scala.util.Random


case class LogisticRegressionHPGroup(hpParameters:Seq[LogisticRegressionHParameter[Double]] = Seq(LRRegParam(), ElasticNet()))
  extends HyperParametersGroup[LogisticRegressionHParameter[Double]] {

  override def mutate(): HyperParametersGroup[LogisticRegressionHParameter[Double]] = {
    LogisticRegressionHPGroup(hpParameters = hpParameters.map(hpModelTpe => hpModelTpe.mutate()))
  }
}

object LogisticRegressionHPGroup {
  val default = LogisticRegressionHPGroup()
}

trait LogisticRegressionHParameter[T <: AnyVal] extends MutableHParameter[T, LogisticRegressionHParameter[T]]

/**
  * https://stackoverflow.com/questions/12182063/how-to-calculate-the-regularization-parameter-in-linear-regression
  *
  * The regularization parameter (lambda) is an input to your model so what you probably want to know is how do you
  * select the value of lambda. The regularization parameter reduces overfitting, which reduces the variance of your
  * estimated regression parameters; however, it does this at the expense of adding bias to your estimate.
  * Increasing lambda results in less overfitting but also greater bias. So the real question is "How much bias are you willing to tolerate in your estimate?"
  *
  * One approach you can take is to randomly subsample your data a number of times and look at the variation in your estimate.
  * Then repeat the process for a slightly larger value of lambda to see how it affects the variability of your estimate.
  * Keep in mind that whatever value of lambda you decide is appropriate for your subsampled data, you can likely use
  * a smaller value to achieve comparable regularization on the full data set.
  */
case class LRRegParam(initialValue: Option[Double] = None) extends LogisticRegressionHParameter[Double] with DoubleHPRange { // we can specialize with Marker trait which parameter can be used with which Model
  override def min: Double = 0.0

  override def max: Double = 1.0

  override def step: Double = 0.1 //TODO can we change step during evolution? we need to detect stagnations/convergence and them change step for fine tuning.

  override def getDefault: Double = round(new Random().nextDouble(), 1) // In theory we are interested not only on round values but on the best ones

  var currentValue: Double = initialValue.getOrElse(getDefault)

  override def mutate(): LRRegParam = {
    val increase = new Random().nextBoolean()  // we might want to jump randomly somewhere but most of the time we need to move slowly
    val newValue = if (increase) Math.min(currentValue + step, max) else Math.max(currentValue - step, min)

    LRRegParam(Some(round(newValue, 1))) // or return this?
  }

  override def toString: String = "lambda:" + currentValue.toString
}

case class ElasticNet(initialValue: Option[Double] = None) extends LogisticRegressionHParameter[Double] with DoubleHPRange { // we can specialize with Marker trait which parameter can be used with which Model
  override def min: Double = 0.0

  override def max: Double = 1.0

  override def step: Double = 0.1 //TODO can we change step during evolution? we need to detect stagnations/convergence and them change step for fine tuning.

  override def getDefault: Double = round(new Random().nextDouble(), 1) // In theory we are interested not only on round values but on the best ones

  var currentValue: Double = initialValue.getOrElse(getDefault)

  override def mutate(): ElasticNet = {
    val increase = new Random().nextBoolean()  // we might want to jump randomly somewhere but most of the time we need to move slowly
    val newValue = if (increase) Math.min(currentValue + step, max) else Math.max(currentValue - step, min)

    ElasticNet(Some(round(newValue, 1)))
  }

  override def toString: String = "elastic_net:" + currentValue.toString
}
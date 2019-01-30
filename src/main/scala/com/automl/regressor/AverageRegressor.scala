package com.automl.regressor

class AverageRegressor extends EnsemblingRegressor[Double]{
  override def evaluate(errors: Seq[Double]): Double = errors.sum / errors.length.toDouble
}

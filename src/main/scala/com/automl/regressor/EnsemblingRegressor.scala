package com.automl.regressor

trait EnsemblingRegressor {
  def evaluate(errors: Seq[Double]): Double
}

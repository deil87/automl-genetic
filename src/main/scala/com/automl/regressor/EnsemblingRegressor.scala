package com.automl.regressor

trait EnsemblingRegressor[T] {
  def evaluate(errors: Seq[T]): T
}

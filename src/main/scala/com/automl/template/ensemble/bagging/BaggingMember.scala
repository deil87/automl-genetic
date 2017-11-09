package com.automl.template.ensemble.bagging

import com.automl.regressor.{AverageRegressor, EnsemblingRegressor}
import com.automl.template.ensemble.EnsemblingMember

trait BaggingMember extends EnsemblingMember {
  override def name: String = "bagging " + super.name

  override def ensemblingRegressor: EnsemblingRegressor = new AverageRegressor()
}


object BaggingMember {

  val simpleBagging = Bagging()
  val poolOfBaggingModels: Set[BaggingMember] = Set(simpleBagging)
}
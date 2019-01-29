package com.automl.template.ensemble.bagging

import com.automl.regressor.{AverageRegressor, EnsemblingRegressor}
import com.automl.template.ensemble.EnsemblingModelMember

trait BaggingMember extends EnsemblingModelMember {
  override def name: String = "bagging " + super.name

  override def ensemblingRegressor: EnsemblingRegressor = new AverageRegressor()
}


object BaggingMember {

  val SimpleBagging = Bagging()
  val poolOfBaggingModels: Set[EnsemblingModelMember] = Set(/*SimpleBagging*/)
}
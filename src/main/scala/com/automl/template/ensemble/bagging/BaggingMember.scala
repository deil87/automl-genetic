package com.automl.template.ensemble.bagging

import com.automl.PaddedLogging
import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.problemtype.ProblemType
import com.automl.regressor.{AverageRegressor, EnsemblingRegressor}
import com.automl.template.ensemble.EnsemblingModelMember

trait BaggingMember extends EnsemblingModelMember { self: PaddedLogging =>
  override def name: String = "Bagging " + super.name

  def ensemblingRegressor[_](problemType: ProblemType): EnsemblingRegressor[_]

}


object BaggingMember {

  val SimpleBagging = SparkGenericBagging()
  val poolOfBaggingModels: Set[EnsemblingModelMember] = Set(SimpleBagging)
}
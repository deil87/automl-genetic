package com.automl.template.ensemble.bagging

import com.automl.classifier.ensemble.bagging.SparkBagging
import com.automl.problemtype.ProblemType
import com.automl.regressor.{AverageRegressor, EnsemblingRegressor}
import com.automl.template.ensemble.EnsemblingModelMember

trait BaggingMember extends EnsemblingModelMember {
  override def name: String = "bagging " + super.name

  def ensemblingRegressor[_](problemType: ProblemType): EnsemblingRegressor[_]

}


object BaggingMember {

  val SimpleBagging = SparkBagging()
  val poolOfBaggingModels: Set[EnsemblingModelMember] = Set(SimpleBagging)
}
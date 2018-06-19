package com.automl.template.ensemble.boosting

import com.automl.template.ensemble.EnsemblingModelMember


trait BoostingMember extends EnsemblingModelMember {
  override def name: String = "boosting " + super.name
}

object BoostingMember {
  val boosting = Boosting()

  val poolOfBoostingModels: Set[BoostingMember] = Set(boosting)

}

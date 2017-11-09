package com.automl.template.ensemble.boosting

import com.automl.template.ensemble.EnsemblingMember


trait BoostingMember extends EnsemblingMember {
  override def name: String = "boosting " + super.name
}

object BoostingMember {
  val boosting = Boosting()

  val poolOfBoostingModels: Set[BoostingMember] = Set(boosting)

}

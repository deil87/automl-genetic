package com.automl.template.ensemble.stacking

import com.automl.template.ensemble.EnsemblingModelMember

trait StackingMember extends EnsemblingModelMember {
  override def name: String = "stacking " + super.name
}

object StackingMember {
  val MyStackingImpl = new GenericStacking()
}

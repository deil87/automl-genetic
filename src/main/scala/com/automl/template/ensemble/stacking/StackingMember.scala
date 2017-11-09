package com.automl.template.ensemble.stacking

import com.automl.template.ensemble.EnsemblingMember

trait StackingMember extends EnsemblingMember {
  override def name: String = "stacking " + super.name
}

object StackingMember {
  val MyStackingImpl = new MyStackingImpl()
}

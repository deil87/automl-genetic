package com.automl.template.ensemble.cascading

import com.automl.template.ensemble.EnsemblingMember

trait CascadingMember extends EnsemblingMember {
  override def name: String = "cascading " + super.name
}

object CascadingMember {
  val MyCascadingImpl = new MyCascadingImpl()
}


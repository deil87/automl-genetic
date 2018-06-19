package com.automl.template.ensemble.cascading

import com.automl.template.ensemble.EnsemblingModelMember

trait CascadingMember extends EnsemblingModelMember {
  override def name: String = "cascading " + super.name
}

object CascadingMember {
  val MyCascadingImpl = new MyCascadingImpl()
}


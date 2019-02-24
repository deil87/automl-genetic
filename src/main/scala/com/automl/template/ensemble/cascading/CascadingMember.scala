package com.automl.template.ensemble.cascading

import com.automl.PaddedLogging
import com.automl.template.ensemble.EnsemblingModelMember

trait CascadingMember extends EnsemblingModelMember { self: PaddedLogging =>
  override def name: String = "cascading " + super.name
}

object CascadingMember {
  val MyCascadingImpl = new MyCascadingImpl()
}


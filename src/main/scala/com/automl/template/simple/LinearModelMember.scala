package com.automl.template.simple

import com.automl.PaddedLogging

/**
  * For now it is just a marker trait for all linear simple models
  */
trait LinearModelMember extends SimpleModelMember { this: PaddedLogging =>
  //TODO look at LinearPerceptronClassifier as well
}



package com.automl.population

import com.automl.evolution.dimension.hparameter.HyperParametersField


class HPPopulation(val individuals: Seq[ HyperParametersField])
  extends Population[HyperParametersField]

package com.automl.benchmark

import scala.util.Random

class MultiRestartBenchmarker[T](numberOfRestarts: Int)(fun: MultiRestartBenchmarker[T] => T) {


  var avgSpecific = 0.0
  var avgBaseline = 0.0
  var countOfSuccesses: Int = 0

  def update(specific: Double, baseline: Double) = {
    avgSpecific += specific
    avgBaseline += baseline
    if( specific > baseline) countOfSuccesses += 1
  }

  for (i <- 0 until numberOfRestarts) {
    val seed = new Random().nextLong()
    fun(this)

  }

  println(" Avg success rate:" + avgSuccessRate)

  def avgSuccessRate = {
    countOfSuccesses.toDouble / numberOfRestarts
  }
}

object MultiRestartBenchmarker {

  def apply[T](numberOfRestarts: Int )(fun: MultiRestartBenchmarker[T] => T): MultiRestartBenchmarker[T] = new MultiRestartBenchmarker[T](numberOfRestarts)(fun)
}
package com.automl.benchmark

import scala.util.Random

class MultiRestartBenchmarker[T](numberOfRestarts: Int, seed: Long)(fun: (MultiRestartBenchmarker[T], Long) => T) {


  private var avgSpecific = 0.0
  private var avgBaseline = 0.0
  private var countOfSuccesses: Int = 0

  val rg = new Random(seed)

  def update(specific: Double, baseline: Double) = {
    avgSpecific += specific
    avgBaseline += baseline
    if( specific > baseline) countOfSuccesses += 1
  }

  for (i <- 0 until numberOfRestarts) {
    val iterationSeed = rg.nextLong()
    fun(this, iterationSeed)
  }

  println(" Avg success rate:" + avgSuccessRate)

  def avgSuccessRate = {
    countOfSuccesses.toDouble / numberOfRestarts
  }

  def avgPerformance: Double = {
    avgSpecific / numberOfRestarts
  }

  def avgBaselinePerformance: Double = {
    avgBaseline / numberOfRestarts
  }
}

object MultiRestartBenchmarker {

  def apply[T](numberOfRestarts: Int, seed: Long )(fun: (MultiRestartBenchmarker[T], Long) => T): MultiRestartBenchmarker[T] =
    new MultiRestartBenchmarker[T](numberOfRestarts, seed)(fun)
}
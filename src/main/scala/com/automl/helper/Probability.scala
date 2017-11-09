package com.automl.helper

case class Probability(value: Double) {

  require(value <= 1 && value >= 0, "Probability only make sense within [0,1]")

  def increaseBy(delta: Double) = {
    if(value + delta > 1) copy(value = 1)
    if(value + delta < 0) copy(value = 0)
    else copy(value = value + delta)
  }

  def <=(other: Double): Boolean = this.value <= other
  def >=(other: Double): Boolean = this.value >= other
  def >(other: Double): Boolean = this.value > other
  def <(other: Double): Boolean = this.value < other

}

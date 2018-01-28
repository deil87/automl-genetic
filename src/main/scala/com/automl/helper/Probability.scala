package com.automl.helper

case class Probability(value: Double) {

  require(value <= 1 && value >= 0, s"Probability only make sense within [0,1]. Attempt to set it to $value")

  def increaseBy(delta: Double): Probability = {
    if(value + delta > 1) copy(value = 1)
    if(value + delta < 0) copy(value = 0)
    else copy(value = value + delta)
  }

  def <=(other: Double): Boolean = this.value <= other
  def >=(other: Double): Boolean = this.value >= other
  def >(other: Double): Boolean = this.value > other
  def <(other: Double): Boolean = this.value < other

}

case class MutationProbabilities(structureProb: Probability, memberProb: Probability) {
  def increaseAllBy(delta: Double): MutationProbabilities = {
    val updated = copy(
      structureProb = structureProb.increaseBy(delta),
      memberProb = memberProb.increaseBy(delta)
    )
    println(s"ProbabilityOfStructMutation increased to: ${updated.structureProb}, ProbabilityOfMemberMutation increased to: ${updated.memberProb}")
    updated
  }
}

object MutationProbabilities {
  def apply(initialProbability: Double): MutationProbabilities = new MutationProbabilities(Probability(initialProbability), Probability(initialProbability))

}

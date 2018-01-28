package com.automl.helper

case class EvolutionTimeBoxesBuilder(timeLimit: Long, numEvolutions: Int, splittingStrategy: Option[TimeBoxesStrategy] = None) {
  def withTimeLimit(timeLimit: Long): EvolutionTimeBoxesBuilder = copy(timeLimit = timeLimit)
  def withNumberOfEvolutions(numEvolutions: Int): EvolutionTimeBoxesBuilder = copy(numEvolutions = numEvolutions)
  def withSplittingStrategy(strategy: TimeBoxesStrategy): EvolutionTimeBoxesBuilder = copy(splittingStrategy = Some(strategy))

  def build: EvolutionTimeBoxes = {
    require(splittingStrategy.isDefined)
    val strategy = splittingStrategy.get

    strategy.timeLimit = timeLimit
    strategy.numRanges = numEvolutions
    EvolutionTimeBoxes(strategy.durations.zipWithIndex.map{ case (duration, idx) => TimeBox(idx, duration)})
  }
}

case class EvolutionTimeBoxes(timeBoxes: Seq[TimeBox])

case class TimeBox(index: Int, duration: Long)

trait TimeBoxesStrategy {
  var timeLimit: Long
  var numRanges: Int
  def durations: Seq[Long]
}

case class EqualEvolutionsStrategy(var timeLimit: Long, var numRanges: Int) extends TimeBoxesStrategy {

  override def durations: Seq[Long] = {
    val step = timeLimit / numRanges
    step to timeLimit by step
  }
}

package com.automl.helper

case class EvolutionTimeBoxesBuilder(globalMaxTime: Long, numEvolutions: Int, splittingStrategy: Option[TimeBoxesStrategy] = None) {
  def withTimeLimit(timeLimit: Long): EvolutionTimeBoxesBuilder = copy(globalMaxTime = timeLimit)
  def withNumberOfEvolutions(numEvolutions: Int): EvolutionTimeBoxesBuilder = copy(numEvolutions = numEvolutions)
  def withSplittingStrategy(strategy: TimeBoxesStrategy): EvolutionTimeBoxesBuilder = copy(splittingStrategy = Some(strategy))

  def build: EvolutionTimeBoxes = {
    require(splittingStrategy.isDefined)
    val strategy = splittingStrategy.get

    strategy.timeLimit = globalMaxTime
    strategy.numRanges = numEvolutions
    EvolutionTimeBoxes(strategy.timeBoundaries.zip(0L +: strategy.timeBoundaries).zipWithIndex.map{
      case ((timeBoundaryCurrent, timeBoundaryPrev), idx) =>
        TimeBox(idx, timeBoundaryPrev, timeBoundaryCurrent, maxTime = globalMaxTime)
    })
  }
}

case class EvolutionTimeBoxes(timeBoxes: Seq[TimeBox])

//Consider renaming. It is actually a time limit for the corresponding evolution since the beginning of the automl. EvolutionTimeLimit?
case class TimeBox(index: Int, lowerBoundary: Long, upperBoundary: Long, maxTime: Long) {
  override def toString: String = s"$index-th TimeBox( ${lowerBoundary / 1000} sec -- ${upperBoundary / 1000} sec )"
}

trait TimeBoxesStrategy {
  var timeLimit: Long
  var numRanges: Int
  def timeBoundaries: Seq[Long]
}

case class EqualEvolutionsStrategy(var timeLimit: Long, var numRanges: Int) extends TimeBoxesStrategy {

  override def timeBoundaries: Seq[Long] = {
    val step = timeLimit / numRanges
    step to timeLimit by step
  }
}

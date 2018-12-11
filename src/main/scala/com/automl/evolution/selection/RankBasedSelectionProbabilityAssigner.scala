package com.automl.evolution.selection

/**
  *   Linear version page 82
  * @tparam T
  */
class RankBasedSelectionProbabilityAssigner[T] extends SelectionProbabilityAssigner[T] {
  override def assign(items: List[T]): List[(T, Double)] = {
    val s = 2
    val m = items.length
    val baselineTerm = (2 - s).toDouble / m
    def prob(rank: Int) = baselineTerm + (2 * rank * ( s - 1)).toDouble /( m * ( m -1) )
    items.zipWithIndex.map{ case( item, rank) =>
      val probabilityForCurrentRank = prob(rank)
      (item, probabilityForCurrentRank)
    }
  }
}

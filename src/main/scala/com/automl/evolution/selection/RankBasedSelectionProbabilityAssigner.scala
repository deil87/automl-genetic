package com.automl.evolution.selection

/**
  *   Linear version page 82
  *   Assign probabilities based on rank. Rank starts with 0 so the last element in the list will have highest probability assigned.
  * @tparam T
  */
class RankBasedSelectionProbabilityAssigner[T] extends SelectionProbabilityAssigner[T] {
  override def assign(items: List[T]): List[(T, Double)] = {
    if (items.length == 1) {
      items.map(item => (item, 1.0))
    } else {
      val s = 2
      val m = items.length
      val baselineTerm = (2 - s).toDouble / m

      def prob(rank: Int) = baselineTerm + (2 * rank * (s - 1)).toDouble / (m * (m - 1))

      items.zipWithIndex.map { case (item, rank) =>
        val probabilityForCurrentRank = prob(rank)
        (item, probabilityForCurrentRank)
      }
    }
  }
}

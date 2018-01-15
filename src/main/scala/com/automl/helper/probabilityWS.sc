val elements = Seq(1, 66, 15, 47, 90)
val sorted = elements.sortWith(_ < _)

val ranked = sorted.zipWithIndex


val s = 1.5
def computeProbabilityLinearRanked(s: Double, populationSize: Long, rank: Long) = {
  ((2 - s) / populationSize) + 2 * rank * (s -1) / (populationSize * (populationSize - 1))
}
val popSize = ranked.length
val linearRanked = ranked.map{ case ( value, rank) => (value, rank, computeProbabilityLinearRanked(s, popSize, rank))}


val list = Seq(1,2,3,4,5)
list.scanLeft(0){case (acc, v) => acc + v }
package com.automl.evolution.evaluation

case class EvaluationContextInfo(workingDataSetSize: Long, evolutionNumber: Int, generationNumber: Int) {
  override def toString: String = s"ev.: $evolutionNumber, gen.: $generationNumber, workingSetSize(tr+t): $workingDataSetSize"
}

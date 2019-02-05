package com.automl.evolution.diversity

import org.apache.spark.sql.DataFrame

abstract class DistanceMetric[DistanceBase] {
    def getDistanceMetrics: DistanceBase
}

abstract class MultidimensionalDistanceMetric extends DistanceMetric[Seq[Double]]{
}

case class MisclassificationDistanceResult(distance: Double, errorsOnSamePositionsCount: Long, matchesOnSamePositionsCount: Long, rawData: DataFrame) extends MultidimensionalDistanceMetric{
//  override def getDistanceMetrics: Seq[Double] = Seq(distance, errorsOnSamePositionsCount + matchesOnSamePositionsCount)
  override def getDistanceMetrics: Seq[Double] = Seq(distance)
}


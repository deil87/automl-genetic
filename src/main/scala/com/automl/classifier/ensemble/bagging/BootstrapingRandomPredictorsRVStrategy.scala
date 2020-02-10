package com.automl.classifier.ensemble.bagging

import com.automl.dataset.{ColumnsSamplingStrategy, RowsSamplingStrategy}
import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.template.{TemplateMember, TemplateTree}
import org.apache.spark.sql.DataFrame

//class BootstrapingRandomPredictorsRVStrategy(samplingStrategy: SamplingStrategy) extends BootstrapingRVStrategy(samplingStrategy = samplingStrategy) {
class BootstrapingRandomPredictorsRVStrategy(rowsSamplingStrategy: RowsSamplingStrategy,
                                             rowsSamplingRatio: Double,
                                             columnsSamplingStrategy: ColumnsSamplingStrategy,
                                             columnsSamplingRatio: Double) extends ReducingVarianceBasedOnDataStrategy {

  override def generateTrainingSamples[A <: TemplateMember](trainDF: DataFrame,
                                                            subMembers: Seq[TemplateTree[A]],
                                                            hyperParamsMap: Option[HyperParametersField],
                                                            seed: Long): Seq[(TemplateTree[A], DataFrame)] = {
    subMembers.zipWithIndex.map { case (member, idx) =>
      val rowsSample: DataFrame = rowsSamplingStrategy.sampleRatio(trainDF, rowsSamplingRatio, seed + idx)
      val colsAndRowsSample = columnsSamplingStrategy.sampleRatio(rowsSample, columnsSamplingRatio, seed, null)
      (member, colsAndRowsSample)
    }
  }

}

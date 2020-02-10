package com.automl.classifier.ensemble.bagging
import com.automl.PaddedLogging
import com.automl.dataset.{SamplingStrategy, StratifiedRowsSampling}
import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.template.{TemplateMember, TemplateTree}
import org.apache.spark.sql.DataFrame

class BootstrapingRVStrategy(samplingStrategy: SamplingStrategy) extends ReducingVarianceBasedOnDataStrategy with PaddedLogging{

  override def generateTrainingSamples[A <: TemplateMember](trainDF: DataFrame,
                                                            subMembers: Seq[TemplateTree[A]],
                                                            hyperParamsMap: Option[HyperParametersField],
                                                            seed: Long): Seq[(TemplateTree[A], DataFrame)] = {
    subMembers.zipWithIndex.map { case (member, idx) =>
      //      val samplingSeed = new Random(seed).nextLong()//seed + idx
      val sample = samplingStrategy.sampleRatio(trainDF, 0.5, seed + idx)
      (member, sample)
    }
  }

  override def logPaddingSize: Int = 0

}

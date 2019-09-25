package com.automl.template.simple

import com.automl.template.TemplateMember
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.DataFrame

trait ClassificationMetricsHelper { self: TemplateMember with LazyLogging =>

  def printConfusionMatrix(predictionsAndLabels: DataFrame, testDF: DataFrame) = {
    import predictionsAndLabels.sparkSession.implicits._
    import utils.SparkMLUtils._
    val metrics = new MulticlassMetrics(predictionsAndLabels.select($"prediction", $"indexedLabel" as "label" ).rdd.map(row => (row.getDouble(0), row.getDouble(1))))

//    testDF.showAll()
    logger.info(s"Confusion matrix for ${this.name} for test split. For CV we need nfolds confusion matrices: \n Rows for actual classes and columns for predicted once.\n" + metrics.confusionMatrix.toString())

  }
}
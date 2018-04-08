package com.automl.template.ensemble.bagging

import com.automl.helper.FitnessResult
import com.automl.regressor.{AverageRegressor, EnsemblingRegressor}
import com.automl.classifier.ensemble.bagging.SparkBagging
import com.automl.template.{EvaluationMagnet, TemplateMember, TemplateTree, TreeContext}
import com.automl.template.ensemble.EnsemblingMember
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.DoubleType
import utils.SparkMLUtils

//TODO maybe we can change NodeTemplate to Bagging/Boosting/Cascading-Template ? we will get access to subMembers then directly
case class Bagging() extends BaggingMember with LazyLogging{
  override def name: String = "Bagging " + super.name


  override def fitnessError(trainDF: DataFrame, testDF: DataFrame): FitnessResult =
    throw new IllegalStateException("We should call ensemblingFitnessError method for ensembling classifiers")

  override def ensemblingFitnessError[A <: TemplateMember](trainDF: DataFrame,
                                                           testDF: DataFrame,
                                                           subMembers: Seq[TemplateTree[A]])
                                                          (implicit tc: TreeContext = TreeContext()): FitnessResult = {
    logger.info(s"\nEvaluating $name ...")
    val sb = new SparkBagging(subMembers)

    val fitness = sb.fitnessError(trainDF, testDF)

    // Validate on testDF
    logger.info(s"$name : fitnessError = " + fitness)
    fitness
  }


  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???


  override def ensemblingRegressor: EnsemblingRegressor = super.ensemblingRegressor
}



package com.automl.template.ensemble.bagging

import com.automl.classifier.ensemble.bagging.SparkBagging
import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.regressor.EnsemblingRegressor
import com.automl.template.{EvaluationMagnet, TemplateMember, TemplateTree, TreeContext}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.sql.DataFrame

//TODO maybe we can change NodeTemplate to Bagging/Boosting/Cascading-Template ? we will get access to subMembers then directly
case class Bagging() extends BaggingMember with LazyLogging{
  override def name: String = "Bagging " + super.name


  override def ensemblingFitnessError[A <: TemplateMember](trainDF: DataFrame,
                                                           testDF: DataFrame,
                                                           subMembers: Seq[TemplateTree[A]],
                                                           problemType: ProblemType)
                                                          (implicit tc: TreeContext = TreeContext()): FitnessResult = {
    logger.info(s"Evaluating $name ...")
    val sb: SparkBagging[A] = new SparkBagging()

    val fitness = sb.ensemblingFitnessError(trainDF, testDF, subMembers, problemType)

    // Validate on testDF
    logger.info(s"$name : fitnessError = " + fitness)
    fitness
  }


  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???


  override def ensemblingRegressor: EnsemblingRegressor = super.ensemblingRegressor
}



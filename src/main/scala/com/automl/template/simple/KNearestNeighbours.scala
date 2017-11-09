package com.automl.template.simple

import com.automl.helper.FitnessResult
import com.automl.template.EvaluationMagnet
import org.apache.spark.sql._

/*
* Could be used for both regression and classification problems.
* There is no implementation in Spark for kNN because the model(data examples) couldn't be efficiently distributed.
* */
case class KNearestNeighbours() extends SimpleModelMember {
  override def name: String = "KNearestNeighbours " + super.name


  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???

  override def fitnessError(trainDF: DataFrame, testDF: DataFrame): FitnessResult = ???

}

package com.automl

import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.automl.template.{EvaluationMagnet, TemplateMember}
import com.automl.template.simple.SimpleModelMember
import org.apache.spark.sql.DataFrame

import scala.util.Random

case class Wildcard(members: Seq[TemplateMember] = SimpleModelMember.poolOfSimpleModels.randSubList)(implicit val logPaddingSize: Int = 0)  extends SimpleModelMember with PaddedLogging {

  lazy val size = members.length


  override def canHandleProblemType: PartialFunction[ProblemType, Boolean] = ???

  override def fitnessError(magnet: EvaluationMagnet): FitnessResult = ???


  override def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType): FitnessResult = {
    throw new IllegalStateException("We can't estimate fitnessError on wildcard for now")
  }

  override def name: String = "Wildcard" + members.map(_.name).mkString("[", ",", "]")

  def materialize: TemplateMember = {
    val chosenMember: TemplateMember = members(Random.nextInt(size))
    chosenMember
  }
  // TODO when to call this method? on new evolution or generation?
  def shrinkDegreeOfFreedom: Wildcard = this.copy(members = members.tail)
}

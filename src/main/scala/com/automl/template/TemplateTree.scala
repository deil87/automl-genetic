package com.automl.template

import java.util.UUID

import com.automl.template.ensemble.EnsemblingModelMember
import com.automl.template.simple.SimpleModelMember
import org.apache.spark.sql.DataFrame
import com.automl.helper.FitnessResult
import com.automl.problemtype.ProblemType
import com.typesafe.scalalogging.LazyLogging
import kamon.Kamon
import kamon.metric.MeasurementUnit


//TODO rename to TemplateTreeNode
sealed trait TemplateTree[+A <: TemplateMember]{

  def member: A
  def subMembers: Seq[TemplateTree[A]]

  def evaluateFitness(trainingDF: DataFrame, testDF: DataFrame, problemType: ProblemType)(implicit tc: TreeContext = TreeContext()): FitnessResult

  def height: Int = 1 + subMembers.foldLeft(1){ case (h, subMember) => Math.max(h, subMember.height)}
}

case class LeafTemplate[+A <: TemplateMember](member: A) extends TemplateTree[A] {
  override def subMembers: Seq[TemplateTree[A]] = throw new UnsupportedOperationException("Leaf template isn't supposed to have subMembers")


  override def evaluateFitness(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType)(implicit tc: TreeContext = TreeContext()): FitnessResult = {

    TemplateTree.updateLeafTC(member.name, height,tc)

    member.fitnessError(trainDF, testDF, problemType)
  }

  override def height: Int = 1
}

//TODO rename node prefix because everything is node. Leaf/non-leaf nodes.
//TODO make upper bound for E type EnsemblingMember
case class NodeTemplate[+A <: TemplateMember](member: A, subMembers: Seq[TemplateTree[A]] = Nil) extends TemplateTree[A] {
  require(member.isInstanceOf[EnsemblingModelMember], "NodeTemplates's member shoud be of ensembling type")

  //We delegating calculation to ensemble member as well
  override def evaluateFitness(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType)(implicit tc: TreeContext = TreeContext()): FitnessResult = {
    val updatedTC = TemplateTree.updateNodeTC(member.name, height, tc)
    //member.fitnessError(trainDF, testDF, subMembers)
    //or
    member.asInstanceOf[EnsemblingModelMember].ensemblingFitnessError(trainDF, testDF, subMembers, problemType)(updatedTC)
  }
}

object TemplateTree extends LazyLogging{

  def updateLeafTC(memberName: String, currentHeight: Int, tc: TreeContext):Unit = {
    val updatedTC = if(tc.level.isDefined) {
      logger.info(s"Evaluating LeafTemplate $memberName on the ${tc.level.get} level")
    }
    else
      tc.copy(level = Some(currentHeight))
  }

  def updateNodeTC(memberName: String, currentHeight: Int, tc: TreeContext):TreeContext = {
    if(tc.level.isDefined) {
      logger.info(s"Evaluating $memberName on the ${tc.level.get} level")
      tc.copy(level = Some(tc.level.get - 1))
    }
    else
      tc.copy(level = Some(currentHeight))
  }
  //TODO redundant
  def firstPopulation = SimpleModelMember.poolOfSimpleModels.map(tm => LeafTemplate(tm))

}

case class TreeContext(level: Option[Int] = None)

trait TemplateMember {
  def name: String

  @Deprecated() //"Consider to remove if there is no way to improve flexibility of evaluation API"
  def fitnessError(magnet: EvaluationMagnet): FitnessResult = null

  //TODO could add TreeContext parameter as well
  //TODO rename to  just fitness
  def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType): FitnessResult

  override def toString: String = name
}

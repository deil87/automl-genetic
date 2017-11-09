package com.automl.template

import com.automl.template.ensemble.EnsemblingMember
import com.automl.template.simple.SimpleModelMember
import org.apache.spark.sql.DataFrame
import com.automl.helper.FitnessResult


sealed trait TemplateTree[+A <: TemplateMember]{
  def member: A
  def subMembers: Seq[TemplateTree[A]]

  def evaluateFitness(trainingDF: DataFrame, testDF: DataFrame)(implicit tc: TreeContext = TreeContext()): FitnessResult

//  def ensembleVisible(implicit evid: A <:< EnsemblingMember) = println("Only for ensembles")

  def height: Int = 1 + subMembers.foldLeft(1){ case (h, subMember) => Math.max(h, subMember.height)}
}

case class LeafTemplate[+A <: TemplateMember](member: A) extends TemplateTree[A] {
  override def subMembers: Seq[TemplateTree[A]] = throw new UnsupportedOperationException("Leaf template isn't supposed to have subMembers")


  override def evaluateFitness(trainDF: DataFrame, testDF: DataFrame)(implicit tc: TreeContext = TreeContext()): FitnessResult = {

    TemplateTree.updateLeafTC(member.name, height,tc)

    member.fitnessError(trainDF, testDF)
  }

  override def height: Int = 1
}

//TODO make upper bound for E type EnsemblingMember
case class NodeTemplate[+A <: TemplateMember](member: A, subMembers: Seq[TemplateTree[A]] = Nil) extends TemplateTree[A] {
  require(member.isInstanceOf[EnsemblingMember], "NodeTemplates's member shoud be of ensembling type")

  //We delegating calculation to ensemble member as well
  override def evaluateFitness(trainDF: DataFrame, testDF: DataFrame)(implicit tc: TreeContext = TreeContext()): FitnessResult = {
    val updatedTC = TemplateTree.updateNodeTC(member.name, height, tc)
    //member.fitnessError(trainDF, testDF, subMembers)
    //or
    member.asInstanceOf[EnsemblingMember].ensemblingFitnessError(trainDF, testDF, subMembers)(updatedTC)
  }
}

object TemplateTree {

  def updateLeafTC(memberName: String, currentHeight: Int, tc: TreeContext):Unit = {
    val updatedTC = if(tc.level.isDefined) {
//      println(s"${List.fill(tc.level.get)("\t").mkString} Evaluating LeafTemplate $memberName on the ${tc.level.get} level")
      println(s"Evaluating LeafTemplate $memberName on the ${tc.level.get} level")
    }
    else
      tc.copy(level = Some(currentHeight))
  }

  def updateNodeTC(memberName: String, currentHeight: Int, tc: TreeContext):TreeContext = {
    if(tc.level.isDefined) {
      println(s"Evaluating $memberName on the ${tc.level.get} level")
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

  def fitnessError(magnet: EvaluationMagnet): FitnessResult

  //TODO could add TreeContext parameter as well
  def fitnessError(trainDF: DataFrame, testDF: DataFrame): FitnessResult

  override def toString: String = name
}

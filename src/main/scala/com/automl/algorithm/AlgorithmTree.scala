package com.automl.algorithm

import com.automl.helper.FitnessResult
import com.automl.template.TemplateMember
import com.automl.template.ensemble.EnsemblingMember
import com.automl.template.simple.SimpleModelMember
import org.apache.spark.sql._

sealed trait AlgorithmTree[+A <: AlgorithmMember]{
  def member: A
  def subMembers: Seq[AlgorithmTree[A]]

  def evaluate(data: DataFrame): FitnessResult

}

case class LeafAlgorithm[A <: AlgorithmMember](member: A) extends AlgorithmTree[A] {
  override def subMembers: Seq[AlgorithmTree[A]] = throw new UnsupportedOperationException("Leaf template isn't supposed to have subMembers")

  override def evaluate(data: DataFrame): FitnessResult = member.evaluateAlgorithmMember(data)
}
case class NodeAlgorithm[A <: AlgorithmMember](member: A, subMembers: Seq[AlgorithmTree[A]] = Nil) extends AlgorithmTree[A] {
  override def evaluate(data: DataFrame): FitnessResult = member match {
    case am: AlgorithmMemberFrom => am.member match {
      case em: EnsemblingMember => ???
      case _ => throw new IllegalStateException("Should be one of hierarchical algorithms")
    }
    case _ => throw new IllegalStateException("Should be one of hierarchical algorithms")
  }
}

trait AlgorithmMember {
  def name: String

  def evaluateAlgorithmMember(data: DataFrame): FitnessResult
}

case class AlgorithmMemberFrom(member: TemplateMember) extends AlgorithmMember{
  override def name: String = member.name

  override def evaluateAlgorithmMember(data: DataFrame): FitnessResult = {
    val Array(trainingSplit,testSplit)  = data.randomSplit(Array(0.67, 0.33),11L)
    member match {
      case bm: SimpleModelMember => bm.fitnessError(trainingSplit, testSplit)
      case em: EnsemblingMember => throw new IllegalStateException("We can evaluate directly only baseModel members")
    }
  }
}

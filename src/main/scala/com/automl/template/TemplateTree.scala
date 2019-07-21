package com.automl.template

import java.util.UUID

import com.automl.PaddedLogging
import com.automl.evolution.dimension.hparameter._
import com.automl.template.ensemble.EnsemblingModelMember
import com.automl.template.simple.{DecisionTree, SimpleModelMember}
import org.apache.spark.sql.DataFrame
import com.automl.helper.{FitnessResult, TemplateTreeHelper}
import com.automl.population.HPPopulation
import com.automl.problemtype.ProblemType
import com.typesafe.scalalogging.LazyLogging
import kamon.Kamon
import kamon.metric.MeasurementUnit
import org.apache.commons.lang3.RandomStringUtils
import scalaz.Lens
import scalaz.std.set

import scala.util.Random


//TODO rename to TemplateTreeNode
sealed trait TemplateTree[+A <: TemplateMember]{

  def member: A

  var logPaddingSize: Int = 0
  def setLogPadding(size: Int): Unit  //We need to set padding first to TemplateTree and then propogate it to members of this tree

  val id: String = RandomStringUtils.randomAlphanumeric(5) // NOTE: Do not use as unique key
//  val orderNumber:Int No we don't want to keep track of the indexes as they will be changing all the time due to evolutions

  def subMembers: Seq[TemplateTree[A]]

  /**
    * @param hyperParamsMap can come from coevolution outside of the TemplateTree or could be present in TemplateTree itself.
    */
  def evaluateFitness(trainingDF: DataFrame, testDF: DataFrame, problemType: ProblemType, hyperParamsMap: Option[HyperParametersField], seed: Long = new Random().nextLong())(implicit tc: TreeContext = TreeContext()): FitnessResult

  var internalHyperParamsMap: Option[HyperParametersField] = Some(HPPopulation.randomHPField)

  def height: Int = 1 + subMembers.foldLeft(1){ case (h, subMember) => Math.max(h, subMember.height)}

  override def equals(obj: Any): Boolean = {
    require(obj.isInstanceOf[TemplateTree[A]])
    val another = obj.asInstanceOf[TemplateTree[A]]
    TemplateTreeComparator.compare(this, another)
  }

  def copy(): TemplateTree[A] = {
//    import scalaz.Scalaz._
//    val streetNumberLens = Lens.lensu[LeafTemplate[TemplateMember], TemplateMember] (
//      (p, memberT) => p.copy(member = memberT),
//      _.member
//    )
    this match {
      case lt: LeafTemplate[A] => {
        LeafTemplate(member = lt.member match {
          case o: DecisionTree => o.copy()(o.logPaddingSize).asInstanceOf[A]
        })
      }
      case lt: NodeTemplate[A] => {
        val nt = NodeTemplate(member = lt.member, subMembers = lt.subMembers)
//        nt.id = lt.id
        val copyOfHPGroups = lt.internalHyperParamsMap.get.modelsHParameterGroups.map{
          case bhpg: BayesianHPGroup  =>
            val smoothing = Smoothing()
            BayesianHPGroup(Seq( smoothing ))//bhpg.copy()
          case bhpg: DecisionTreeHPGroup  => DecisionTreeHPGroup()//bhpg.copy()
          case bhpg: LogisticRegressionHPGroup  => LogisticRegressionHPGroup()//bhpg.copy()
          case unknown  => throw new IllegalStateException(s"Method copy failed due to unsupported copy for $unknown")
        }
        nt.internalHyperParamsMap = Some(lt.internalHyperParamsMap.get.copy(modelsHParameterGroups = copyOfHPGroups))
        nt
      }
      case _ => throw new IllegalStateException()
    }
  }
}

case class LeafTemplate[+A <: TemplateMember](member: A) extends TemplateTree[A] {
  override def subMembers: Seq[TemplateTree[A]] = throw new UnsupportedOperationException("Leaf template isn't supposed to have subMembers")


  override def setLogPadding(size: Int): Unit = member.setLogPadding(size)

  override def evaluateFitness(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType, hyperParamsMap: Option[HyperParametersField], seed: Long)(implicit tc: TreeContext = TreeContext()): FitnessResult = {

    TemplateTree.updateLeafTC(member.name, height,tc)(logPaddingSize)

    trainDF.cache()
    testDF.cache()
    member.fitnessError(trainDF, testDF, problemType)
  }

  override def height: Int = 1
}

//TODO rename node prefix because everything is node. Leaf/non-leaf nodes.
//TODO make upper bound for E type EnsemblingMember
case class NodeTemplate[+A <: TemplateMember](member: A, subMembers: Seq[TemplateTree[A]] = Nil) extends TemplateTree[A] {
  require(member.isInstanceOf[EnsemblingModelMember], "NodeTemplates's member shoud be of ensembling type")

  override def setLogPadding(size: Int): Unit = {
    member.setLogPadding(size)
    subMembers.foreach(_.setLogPadding(size + 4))
  }

  //We delegating calculation to ensemble member as well
  override def evaluateFitness(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType, hyperParamsField: Option[HyperParametersField], seed: Long)(implicit tc: TreeContext = TreeContext()): FitnessResult = {
    val updatedTC = TemplateTree.updateNodeTC(member.name, height, tc)(logPaddingSize)
    //member.fitnessError(trainDF, testDF, subMembers)
    //or
    member.asInstanceOf[EnsemblingModelMember].ensemblingFitnessError(trainDF, testDF, subMembers, problemType, hyperParamsField, seed)(updatedTC)
  }
}

object TemplateTree extends PaddedLogging {

  override def logPaddingSize: Int = 0

  def updateLeafTC(memberName: String, currentHeight: Int, tc: TreeContext)(implicit logPaddingSize: Int):Unit = {
    val updatedTC = if(tc.level.isDefined) {
      debugImpl(s"Evaluating LeafTemplate $memberName on the ${tc.level.get} level")
    }
    else
      tc.copy(level = Some(currentHeight))
  }

  def updateNodeTC(memberName: String, currentHeight: Int, tc: TreeContext)(implicit logPaddingSize: Int):TreeContext = {
    if(tc.level.isDefined) {
      debugImpl(s"Evaluating $memberName on the ${tc.level.get} level")
      tc.copy(level = Some(tc.level.get - 1))
    }
    else
      tc.copy(level = Some(currentHeight))
  }

  implicit def helper(templateTree: TemplateTree[TemplateMember]) = new {
    def render: String = TemplateTreeHelper.renderAsString_v2(templateTree)

    //it is kind of NodeTemplate traversal as we don't have function to check on Leaf nodes
    def traverseAndCheck(fun: TemplateTree[_] => Boolean): Boolean = {
      def traverseAndCheck(template: TemplateTree[_], fun: TemplateTree[_] => Boolean):Boolean = template match {
        case lt@LeafTemplate(_) => true
        case nt@NodeTemplate(_, subMembers) => fun(nt) && subMembers.forall(t => traverseAndCheck(t, fun))
      }
      traverseAndCheck(templateTree, fun)
    }
  }

}

case class TreeContext(level: Option[Int] = None)

trait TemplateMember { self: PaddedLogging =>
  def name: String

  @Deprecated() //"Consider to remove if there is no way to improve flexibility of evaluation API"
  def fitnessError(magnet: EvaluationMagnet): FitnessResult = null

  //TODO could add TreeContext parameter as well
  //TODO rename to  just fitness
  def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType): FitnessResult

  override def toString: String = name

  def setLogPadding(size: Int): Unit = {
    overrideTo = size
  }
}

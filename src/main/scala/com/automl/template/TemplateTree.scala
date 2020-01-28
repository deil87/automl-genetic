package com.automl.template

import java.util.UUID

import com.automl.{ConfigProvider, EvaluationRules, PaddedLogging}
import com.automl.evolution.dimension.hparameter._
import com.automl.template.ensemble.EnsemblingModelMember
import com.automl.template.simple.{DecisionTree, SimpleModelMember}
import org.apache.spark.sql.DataFrame
import com.automl.helper.{FitnessResult, TemplateTreeHelper}
import com.automl.population.HPPopulation
import com.automl.problemtype.ProblemType
import com.typesafe.config.Config
import com.typesafe.scalalogging.LazyLogging
import kamon.Kamon
import kamon.metric.MeasurementUnit
import org.apache.commons.lang3.RandomStringUtils

import scala.util.Random


//TODO rename to TemplateTreeNode
sealed trait TemplateTree[+A <: TemplateMember]{

  def member: A

  var logPaddingSize: Int = 0
  def setLogPadding(size: Int): Unit  //We need to set padding first to TemplateTree and then propogate it to members of this tree

  val id: String = RandomStringUtils.randomAlphanumeric(5) // NOTE: Do not use as unique key
//  val orderNumber:Int No we don't want to keep track of the indexes as they will be changing all the time due to evolutions

  def subMembers: Seq[TemplateTree[A]]

  var parent: Option[NodeTemplate[TemplateMember]] = None

  /**
    * @param hpFieldFromCoevolution come from coevolution outside of the TemplateTree
    */
  def evaluateFitness(trainingDF: DataFrame,
                      testDF: DataFrame,
                      problemType: ProblemType,
                      hpFieldFromCoevolution: Option[HyperParametersField],
                      seed: Long = new Random().nextLong())
                     (implicit tc: TreeContext = TreeContext()): FitnessResult

  // TODO is it right that TemplateTree has HPField/HPGroups, when it is TemplateMember who needs it?
  // This HP map will serve for all nodes of the templateTree. If we need ability to specify  on per member basis
//  var internalHyperParamsMap = Some(HPPopulation.randomRelevantHPFieldFor(member))
  // TODO we should not set HPField to every level of the tree. We nee to set HPField to root node and then share it's instance down the tree.
  //TODO remove
  var internalHyperParamsMap: Option[HyperParametersField] = Some(HPPopulation.randomRelevantHPFieldFor(member)) // ref to root HPField

//  var internalHyperParamsMap: Option[HyperParametersField] = if (ConfigProvider.config.getBoolean("evolution.hyperParameterDimension.enabled"))
//    Some(HPPopulation.randomRelevantHPFieldFor(member))
//  else None

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
       /* val copyOfHPGroups = lt.internalHyperParamsMap.get.modelsHParameterGroups.map{
          case bhpg: BayesianHPGroup  =>
            val smoothing = Smoothing() //TODO fixme can't do this as it will give random hps
            BayesianHPGroup(Seq( smoothing ))//bhpg.copy()
          case bhpg: DecisionTreeHPGroup  => DecisionTreeHPGroup()//bhpg.copy()
          case bhpg: LogisticRegressionHPGroup  => LogisticRegressionHPGroup()//bhpg.copy()
          case bhpg: RandomForestHPGroup  => RandomForestHPGroup()//bhpg.copy()
          case unknown  => throw new IllegalStateException(s"Method copy failed due to unsupported copy for $unknown")
        }
        nt.internalHyperParamsMap = Some(lt.internalHyperParamsMap.get.copy(modelsHParameterGroups = copyOfHPGroups))*/
        nt
      }
      case _ => throw new IllegalStateException()
    }
  }

  override def toString: String = TemplateTreeHelper.renderAsString_v2(this)
}

case class LeafTemplate[+A <: TemplateMember](member: A) extends TemplateTree[A] {
  override def subMembers: Seq[TemplateTree[A]] = throw new UnsupportedOperationException("Leaf template isn't supposed to have subMembers")

  override def setLogPadding(size: Int): Unit = member.setLogPadding(size)

  override def evaluateFitness(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType, hyperParamsField: Option[HyperParametersField], seed: Long)(implicit tc: TreeContext = TreeContext()): FitnessResult = {

    TemplateTree.updateLeafTC(member.name, height,tc)(logPaddingSize)

    trainDF.cache()
    testDF.cache()

    member.fitnessErrorWithValidation(trainDF, testDF, problemType, hyperParamsField)
  }

  override def height: Int = 1
}

//TODO rename node prefix because everything is node. Leaf/non-leaf nodes.
//TODO make upper bound for E type EnsemblingMember
case class NodeTemplate[+A <: TemplateMember](member: A, subMembers: Seq[TemplateTree[A]] = Nil) extends TemplateTree[A] {
  require(member.isInstanceOf[EnsemblingModelMember], "NodeTemplates's member shoud be of ensembling type")

  // Constructor: assigning parents to submembers
  subMembers.foreach(subMember => subMember.parent = Some(this))

  override def setLogPadding(size: Int): Unit = {
    member.setLogPadding(size)
    subMembers.foreach(_.setLogPadding(size + 4))
  }

  //We delegating calculation to ensemble member as well
  override def evaluateFitness(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType, hpFieldFromCoevolution: Option[HyperParametersField], seed: Long)(implicit tc: TreeContext = TreeContext()): FitnessResult = {
    val updatedTC = TemplateTree.updateNodeTC(member.name, height, tc)(logPaddingSize)
    //member.fitnessError(trainDF, testDF, subMembers)
    //or
    member.asInstanceOf[EnsemblingModelMember].ensemblingFitnessErrorWithValidation(trainDF, testDF, subMembers, problemType, hpFieldFromCoevolution, seed)(updatedTC)
  }

  var degreeOfExploration: Double = 0
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

// TODO do we need SimpleModelMember ? it could probably be just TemplateMember. Or maybe marker class is fine...hm
trait TemplateMember extends EvaluationRules { self: PaddedLogging =>
  def name: String

  // We can have a real type of HyperParametersGroup in subclasses if we introduce type parameter for TemplateMember
  var hpGroupInternal: HyperParametersGroup[_ <: MutableHParameter[Double, _]]

  def canHandleProblemType: PartialFunction[ProblemType, Boolean]

  @Deprecated() //"Consider to remove if there is no way to improve flexibility of evaluation API"
  def fitnessError(magnet: EvaluationMagnet): FitnessResult = null

  //TODO could add TreeContext parameter as well
  //TODO rename to  just fitness
  def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType): FitnessResult = {
    val msg = "Consider to use fitnessError() method with `hyperParametersField` parameter"
    info(s"!!!!!!!!!!!! $msg")
    throw new IllegalStateException(msg)
    fitnessErrorWithValidation(trainDF, testDF, problemType, None)
  }

  def fitnessErrorWithValidation(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType, hpFieldFromCoevolution: Option[HyperParametersField] = None): FitnessResult = {
    require(canHandleProblemType(problemType), "canHandleProblemType requirement failed")
    fitnessError(trainDF, testDF, problemType, hpFieldFromCoevolution)
  }

  def fitnessError(trainDF: DataFrame, testDF: DataFrame, problemType: ProblemType, hpFieldFromCoevolution: Option[HyperParametersField] = None): FitnessResult

  /**
    * Selects hpGroup based on the way how we decided to derive it. Either from coevolution or through regular mutation from TemplateTree
    */
  //TODO !!!! If we make Members to provide it's own HPGroups we don't need to select between two hp fields ( i.e. from Coevolutin or default random field)
  def getRelevantHPGroupFromActiveHPField( hpFieldFromCoevolution: Option[HyperParametersField]): Option[HyperParametersGroup[_]] = {

    hpFieldFromCoevolution
      .flatMap(_.modelsHParameterGroups.find(_.isRelevantTo(this)))
//      .orElse(throw new IllegalStateException("getRelevantHPGroupFromActiveHPField should not be called when hpCoevolution is disabled"))
  }

  override def equals(obj: Any): Boolean = {
    require(obj.isInstanceOf[TemplateMember])
    val another = obj.asInstanceOf[TemplateMember]
    (this.hpGroupInternal == another.hpGroupInternal) &&
      this.name == another.name &&
      this.getClass == another.getClass // probably unnecessary as hps are typed
  }

  override def toString: String = name

  def setLogPadding(size: Int): Unit = {
    overrideTo = size
  }
}

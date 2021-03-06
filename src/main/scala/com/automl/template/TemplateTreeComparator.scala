package com.automl.template

import com.typesafe.scalalogging.LazyLogging

object TemplateTreeComparator extends LazyLogging{


  def compare[A <: TemplateMember](left: TemplateTree[A], right: TemplateTree[A]): Boolean = {
    left match {
      case l: LeafTemplate[A] => {
//        val hpFieldsAreBothEmpty = l.internalHyperParamsMap.isEmpty && right.internalHyperParamsMap.isEmpty
//        val hpFieldsAreDefined = l.internalHyperParamsMap.isDefined && right.internalHyperParamsMap.isDefined
//          if(!hpFieldsAreDefined) println("hpFieldsAreDefined are not defined in both trees")
//        lazy val hpFieldsAreEqual = hpFieldsAreDefined && l.internalHyperParamsMap.get == right.internalHyperParamsMap.get
//          if(!hpFieldsAreEqual) logger.debug("Hyperparameter fields of the LeafTemplates are not equal")
        val membersAreEqual = l.member == right.member
//        (hpFieldsAreEqual || hpFieldsAreBothEmpty) && membersAreEqual
        membersAreEqual
      }
      case l: NodeTemplate[A] if right.isInstanceOf[NodeTemplate[A]] => {
        calculateDiffAndCheck(l, right)
      }
      case _ => false
    }
  }

  private def calculateDiffAndCheck[A <: TemplateMember](left: TemplateTree[A], right: TemplateTree[A]): Boolean = {
    val leftMembersSet = left.subMembers.toSet
    val rightMembersSet = right.subMembers.toSet
    val subtractionResult = (leftMembersSet /: rightMembersSet) { case (set, elem) =>
      if (set.contains(elem)) set - elem
      else {
        return false
      }
    }
    subtractionResult  == Set()
  }
}

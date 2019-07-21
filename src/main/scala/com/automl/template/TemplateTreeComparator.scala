package com.automl.template

object TemplateTreeComparator {


  def compare[A <: TemplateMember](left: TemplateTree[A], right: TemplateTree[A]): Boolean = {
    left match {
      case l: LeafTemplate[A] => {
        val hpFieldsAreDefined = l.internalHyperParamsMap.isDefined && right.internalHyperParamsMap.isDefined
        val hpFieldsAreEqual = l.internalHyperParamsMap.get == right.internalHyperParamsMap.get
        val membersAreEqual = l.member == right.member
        membersAreEqual && hpFieldsAreDefined && hpFieldsAreEqual
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

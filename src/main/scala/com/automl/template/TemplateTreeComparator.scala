package com.automl.template

object TemplateTreeComparator {


  def compare[A <: TemplateMember](left: TemplateTree[A], right: TemplateTree[A]): Boolean = {
    left match {
      case l: LeafTemplate[A] => {
        l.member == right.member
      }
      case l: NodeTemplate[A] if right.isInstanceOf[NodeTemplate[A]] => {
        calculateDiffAndCheck(l, right)
      }
      case _ => false
    }
  }

  private def calculateDiffAndCheck[A <: TemplateMember](left: TemplateTree[A], right: TemplateTree[A]): Boolean = {
    val subtractionResult = (left.subMembers.toSet /: right.subMembers.toSet.seq) { case (set, elem) =>
      if (set.contains(elem)) set - elem
      else {
        return false
      }
    }
    subtractionResult  == Set()
  }
}

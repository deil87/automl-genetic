package com.automl.helper

import com.automl.Wildcard
import com.automl.template.simple.SimpleModelMember
import com.automl.template.{LeafTemplate, NodeTemplate, TemplateMember, TemplateTree}

import scala.collection.mutable.Queue

object TemplateTreeHelper {


  def traverse(template: TemplateTree[TemplateMember]): Queue[TemplateMember] = {
    def recursive(template: TemplateTree[TemplateMember], acc: Queue[TemplateMember]): Queue[TemplateMember] = template match {
      case LeafTemplate(x) =>
        acc.enqueue(x)
        acc
      case NodeTemplate(x, subMembers) =>
        acc.enqueue(x)
        subMembers.foldLeft(acc){case (ac, subMember) => recursive(subMember, ac)}
    }


    val queue = new scala.collection.mutable.Queue[TemplateMember]()
    recursive(template, queue)
  }

  def traverse2(template: TemplateTree[TemplateMember]): Queue[String] = {
    def recursive(prefix: String = "", childrenPrefix: String = "", template: TemplateTree[TemplateMember], acc: Queue[String]): Queue[String] = template match {
      case lt@LeafTemplate(x) =>
        val hpRepresentation = if(lt.member.hpGroupInternal == null) "--" else lt.member.hpGroupInternal.toString
        acc.enqueue(/*"L" + */prefix + x.name + " hps: " + hpRepresentation + " hasParent:" + lt.parent.isDefined )
        acc
      case nt@NodeTemplate(x, subMembers) =>
        val hpRepresentation = nt.member.hpGroupInternal.toString // TODO FIXME need to include hash of sumbembers as well
        acc.enqueue(/*"N" + */prefix + x.name + " hps: " + hpRepresentation + " hasParent:" + nt.parent.isDefined + " doe:" + nt.degreeOfExploration )

        subMembers.zipWithIndex.foldLeft(acc){case (ac, (subMember, i)) =>
          if (subMembers.size - 1 != i) {
            recursive(childrenPrefix + "├── ", childrenPrefix + "|   ", subMember, ac)
          } else {
            recursive(childrenPrefix + "└── ", childrenPrefix + "    ", subMember, ac)
          }
        }
        acc
    }


    val queue = new scala.collection.mutable.Queue[String]()
    recursive("", "", template, queue)
  }

  def materialize(template: TemplateTree[TemplateMember]): TemplateTree[TemplateMember] = {

    def materializeTemplateMember(tm: TemplateMember): TemplateMember = tm match {
      case wc@Wildcard(set) => wc.materialize
      case m => m
    }
    template match {
      case LeafTemplate(x) =>
        LeafTemplate(materializeTemplateMember(x).asInstanceOf[SimpleModelMember])

      case NodeTemplate(x, subMembers) =>
        NodeTemplate(materializeTemplateMember(x), subMembers.map(sm => materialize(sm)))
    }

  }

  def renderAsString(template: TemplateTree[TemplateMember]): String = {
    traverse(template).map(_.name).mkString(",")
  }

  def renderAsString_v2(template: TemplateTree[TemplateMember]): String = {
    traverse2(template).mkString("\n")
  }

}

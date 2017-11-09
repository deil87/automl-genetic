package com.automl.helper

import com.automl.Wildcard
import com.automl.algorithm._
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
    def recursive(template: TemplateTree[TemplateMember], acc: Queue[String]): Queue[String] = template match {
      case LeafTemplate(x) =>
        acc.enqueue(x.name)
        acc
      case NodeTemplate(x, subMembers) =>
        acc.enqueue(x.name)
        acc.enqueue("{")
        subMembers.zipWithIndex.foldLeft(acc){case (ac, (subMember, i)) =>
          recursive(subMember, ac)
            if(subMembers.size - 1 != i)
              ac.enqueue(",")
            ac
        }

        acc.enqueue("}")
        acc
    }


    val stack = new scala.collection.mutable.Queue[String]()
    recursive(template, stack)
  }

  def materialize(template: TemplateTree[TemplateMember]): AlgorithmTree[AlgorithmMember] = {

    def materializeTemplateMember(tm: TemplateMember): AlgorithmMember = tm match {
      case wc@Wildcard(set) => wc.materialize
      case m => AlgorithmMemberFrom(m)
    }
    template match {
      case LeafTemplate(x) =>
        LeafAlgorithm(materializeTemplateMember(x))

      case NodeTemplate(x, subMembers) =>
        NodeAlgorithm(materializeTemplateMember(x), subMembers.map(sm => materialize(sm)))
    }

  }

  def print(template: TemplateTree[TemplateMember]): String = {
    traverse(template).map(_.name).mkString(",")
  }

  def print2(template: TemplateTree[TemplateMember]): String = {
    traverse2(template).mkString("")
  }
}

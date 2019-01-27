package com.automl.evolution.diversity

import java.util

import pal.tree.{SimpleNode, SimpleTree, UnconstrainedTree}
import treecmp.metric.TripletMetricSimple

import scala.collection.JavaConverters._

class TripletDistanceWrapper {


  def computeDistance = {
    lazy val children = Seq(
      new SimpleNode("A", 1),
      new SimpleNode("B", 1),
      new SimpleNode("C", 1))
      .asJava.toArray().asInstanceOf[Array[SimpleNode]]

    val node = new SimpleNode("A", 42)
    val treeOne = new SimpleTree(node)

    println(treeOne.toString)
    new TripletMetricSimple().getDistForArbitrary(treeOne, treeOne)
  }
}

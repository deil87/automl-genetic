package com.automl.helper

import com.automl.template.ensemble.EnsemblingModelMember
import com.automl.template.simple.SimpleModelMember
import com.automl.template.{TemplateMember, TemplateTree}

import scala.collection.mutable

case class Probability(value: Double) {

  require(value <= 1 && value >= 0, s"Probability only make sense within [0,1]. Attempt to set it to $value")

  def increaseBy(delta: Double): Probability = {
    if(value + delta > 1) copy(value = 1)
    if(value + delta < 0) copy(value = 0)
    else copy(value = value + delta)
  }

  def <=(other: Double): Boolean = this.value <= other
  def >=(other: Double): Boolean = this.value >= other
  def >(other: Double): Boolean = this.value > other
  def <(other: Double): Boolean = this.value < other

}

object Probability {

  implicit val probNum = new Numeric[Probability]{
    override def plus(x: Probability, y: Probability): Probability = ???

    override def minus(x: Probability, y: Probability): Probability = ???

    override def times(x: Probability, y: Probability): Probability = ???

    override def negate(x: Probability): Probability = ???

    override def fromInt(x: Int): Probability = ???

    override def toInt(x: Probability): Int = ???

    override def toLong(x: Probability): Long = ???

    override def toFloat(x: Probability): Float = ???

    override def toDouble(x: Probability): Double = ???

    override def compare(x: Probability, y: Probability): Int = ???
  }
}


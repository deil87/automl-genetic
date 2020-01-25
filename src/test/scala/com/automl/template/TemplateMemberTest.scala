package com.automl.template

import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.evolution.dimension.hparameter._
import com.automl.template.simple.{DecisionTree, LogisticRegressionModel}
import org.bytedeco.javacpp.opencv_ml.LogisticRegression
import org.scalatest.{FunSuite, Matchers}

import scala.collection.mutable

class TemplateMemberTest  extends FunSuite with Matchers  {

  test ("two TemplateMembers with same hps should be equal") {

    val dtTemplate = DecisionTree(DecisionTreeHPGroup(Seq(MaxDepth(Some(4.0)))))
    val dtTemplate2 = DecisionTree(DecisionTreeHPGroup(Seq(MaxDepth(Some(4.0)))))

    dtTemplate shouldEqual dtTemplate2
  }

  test ("two TemplateMembers with different hps should not be equal") {

    val dtTemplate = DecisionTree(DecisionTreeHPGroup(Seq(MaxDepth(Some(4.0)))))
    val dtTemplate2 = DecisionTree(DecisionTreeHPGroup(Seq(MaxDepth(Some(5.0)))))

    dtTemplate == dtTemplate2 shouldBe false
  }

  test ("we can set hpGroups") {

    val lr = LogisticRegressionModel()
    val lrHPGroup = LogisticRegressionHPGroup()
    val hp: HyperParametersGroup[_ <: MutableHParameter[Double, _]] = lr.hpGroupInternal
    lr.hpGroupInternal = lrHPGroup
  }

  test ("we can set hpGroups for Bagging") {

//    val lr = SparkGenericBagging()
//    val lrHPGroup = BaggingHPGroup()
//    val res: HyperParametersGroup[BaggingHParameter] = lrHPGroup.mutate()
//    val hp: HyperParametersGroup[_ <: MutableHParameter[Double, _]] = lr.hpGroupInternal
//    hp should not be lrHPGroup


  }

  /*test ("we can set hpGroups for dummy classes") {

    val monday = Monday()
    val tuesday = Tuesday[Int]()
    val weekend = Weekend(tuesday)
    val res: Tuesday[Int] = weekend.getDay

  }*/

  test ("we can set hpGroups for dummy classes") {

//    val monday = Monday()
//    val tuesday = Tuesday[Int]()
//    val week = MondayWeek()
    val week2 = TuesdayWeek()
    val res: Tuesday[Int] = week2.getWek

    var hpG:Weekend[_ <: Day[Int, _]] = week2

    var hpG2:Weekend[_ <: Day[Int, _]] = week2


//    hpG should not be hpG2
  }
}


/*trait Day[-T, M <: Day[T, M]] // making T contrvariant -T makes Day to accept K<:Int

case class Monday() extends Day[Int, Monday]
case class Tuesday[K <: Int]() extends Day[K, Monday]

case class Weekend[D <: Day[Int, Monday]](day: D) {
  def getDay:D = day
}*/




sealed trait TimeRange[+T] {
  def getDefault:T
}

trait Day[T, M <: Day[T, M]] extends TimeRange[T] { // like trait MutableHParameter[T, V <: MutableHParameter[T, V]] extends HParameter[T] {
  def currentValue: M
}

//case class Monday() extends Day[Int, Monday] {
//
//  override def currentValue: Monday = ???
//
//  override def getDefault: Int = ???
//
//}

trait Tuesday[K <: AnyVal] extends Day[K, Tuesday[K]] { // like trait LogisticRegressionHParameter[T <: AnyVal] extends MutableHParameter[T, LogisticRegressionHParameter[T]]

  override def currentValue: Tuesday[K] = ???

  override def getDefault: K = ???
}

case class TuesdayImpl() extends Tuesday[Int]

trait Weekend[D <: Day[Int, D]] {  // like trait HyperParametersGroup[HPModelBoundedType <: MutableHParameter[Double, HPModelBoundedType]]{
  def getWek: D
}

//case class MondayWeek() extends Weekend[Monday] { // like LogisticRegressionHPGroup
//  override def getWek: Monday = Monday()
//}

case class TuesdayWeek() extends Weekend[Tuesday[Int]] { // like LogisticRegressionHPGroup extends HyperParametersGroup[LogisticRegressionHParameter[Double]]
  override def getWek: Tuesday[Int] = TuesdayImpl()
}


/*

sealed trait TimeRange[+T] {
  def getDefault[U >: T]:U
}

trait Day[+T, M <: Day[T, M]] extends TimeRange[T] {
  def currentValue[U >: T]: U
}

case class Monday() extends Day[Int, Monday] {
  override def currentValue[U >: Int]: U = ???

  override def getDefault[U >: Int]: U = ???
}

case class Tuesday[K <: Int]() extends Day[K, Monday] {
  override def currentValue[U >: K]: U = ???

  override def getDefault[U >: K]: U = ???
}

trait Weekend[D <: Day[Int, D]] {  // analog HyperParametersGroup
  def getWek: D
}

case class MondayWeek() extends Weekend[Monday] {
  override def getWek: Monday = ???
}*/

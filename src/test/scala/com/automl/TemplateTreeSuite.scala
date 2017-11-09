package com.automl

import com.automl.helper.TemplateTreeHelper
import com.automl.template._
import com.automl.template.simple.{Bayesian, LinearRegressionModel, GradientBoosting, RandomForest}
import org.scalatest.{FunSuite, Matchers}


class TemplateTreeSuite extends FunSuite with Matchers{

  /*test("Generation of Individual members tree") {

    val individual = TemplateTree.generate(3, 2)

    assert(Option(individual).isDefined)
  }*/

  val individual =
    NodeTemplate(new Bayesian(),
      Seq(
        LeafTemplate(new LinearRegressionModel()),
        NodeTemplate(new RandomForest(),
          Seq(
            LeafTemplate(new GradientBoosting())
          )
        )
      )
    )

  test("TemplateMemberHelper should traverse tree") {
    val stringRepr = TemplateTreeHelper.print(individual)
    val margin: String =
      """Bayesian baseModel member,
          |GenerilizedLinearModel baseModel member,
          |Random forest bagging ensembling member,
            |Gradient boosting ensembling member"""
        .stripMargin.replaceAll("\n","")
    assert(stringRepr == margin)
  }

  test("TemplateMemberHelper should print tree") {
    val stringRepr: String = TemplateTreeHelper.print2(individual)
    val margin: String =
      """Bayesian baseModel member{GenerilizedLinearModel baseModel member,Random forest bagging ensembling member{Gradient boosting ensembling member}}"""
        .stripMargin.replaceAll("\n","")
    assert(stringRepr == margin)
  }

  /*test("Generation of random tree") {

    val individual = TemplateTree.generate(3, 2)
    println(TemplateTreeHelper.print2(individual))

    assert(Option(individual).isDefined)
  }*/


  test("TemplateMemberHelper should print tree with wildcards") {
    val individualWithWildcard =
      NodeTemplate(new Bayesian(),
        Seq(
          LeafTemplate(new LinearRegressionModel()),
          NodeTemplate(new RandomForest(),
            Seq(
              LeafTemplate(Wildcard(List(new RandomForest(), new Bayesian())))
            )
          )
        )
      )
    val stringRepr: String = TemplateTreeHelper.print2(individualWithWildcard)
    val margin: String =
      """Bayesian baseModel member{GenerilizedLinearModel baseModel member,Random forest bagging ensembling member{Wildcard[Random forest bagging ensembling member,Bayesian baseModel member]}}"""
        .stripMargin.replaceAll("\n","")
    assert(stringRepr == margin)
  }

}


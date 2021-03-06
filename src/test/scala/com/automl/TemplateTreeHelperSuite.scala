package com.automl

import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.helper.TemplateTreeHelper
import com.automl.template._
import com.automl.template.ensemble.stacking.GenericStacking
import com.automl.template.simple.{Bayesian, GradientBoosting, LinearRegressionModel, RandomForest}
import org.scalatest.{FunSuite, Matchers}


class TemplateTreeHelperSuite extends FunSuite with Matchers{

  val individual =
    NodeTemplate(SparkGenericBagging(),
      Seq(
        LeafTemplate(new LinearRegressionModel()),
        LeafTemplate(new Bayesian()),
        NodeTemplate(new GenericStacking(),
          Seq(
            LeafTemplate(new GradientBoosting()),
            NodeTemplate(new GenericStacking(),
              Seq(
                LeafTemplate(new GradientBoosting())
              )
            )
          )
        ),
        LeafTemplate(new LinearRegressionModel()),
        LeafTemplate(new Bayesian())
      )
    )

  test("TemplateTreeHelper should print tree") {
    val stringRepr: String = TemplateTreeHelper.renderAsString_v2(individual)
    println(stringRepr)
    assert(stringRepr.sliding("└──".length).count(_ == "└──") == 3)
  }

  ignore("TemplateMemberHelper should traverse tree") {
    val stringRepr = TemplateTreeHelper.renderAsString(individual)
    val margin: String =
      """Bayesian baseModel member,
          |GenerilizedLinearModel baseModel member,
          |Random forest bagging ensembling member,
            |Gradient boosting ensembling member"""
        .stripMargin.replaceAll("\n","")
    assert(stringRepr == margin)
  }

  ignore("TemplateMemberHelper should print tree with wildcards") {
    val individualWithWildcard =
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(LinearRegressionModel()),
          LeafTemplate(Bayesian()),
          NodeTemplate(SparkGenericBagging(),
            Seq(
              LeafTemplate(Wildcard(List(RandomForest(), Bayesian())))
            )
          )
        )
      )
    val stringRepr: String = TemplateTreeHelper.renderAsString_v2(individualWithWildcard)
    val margin: String =
      """Bayesian baseModel member{GenerilizedLinearModel baseModel member,Random forest bagging ensembling member{Wildcard[Random forest bagging ensembling member,Bayesian baseModel member]}}"""
        .stripMargin.replaceAll("\n","")
    assert(stringRepr == margin)
  }

}


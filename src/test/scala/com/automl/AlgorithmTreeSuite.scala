package com.automl

import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.helper.TemplateTreeHelper
import com.automl.template._
import com.automl.template.simple._
import org.scalatest.{FunSuite, Matchers}
import com.test.TestBase


class AlgorithmTreeSuite extends FunSuite with TestBase{


  val template =
    NodeTemplate(SparkGenericBagging(),
      Seq(
        LeafTemplate(LinearRegressionModel()),
        NodeTemplate(SparkGenericBagging(),
          Seq(
            LeafTemplate(Wildcard(List(Bayesian())))
          )
        )
      )
    )


  ignore("TemplateMemberHelper should materialize to algorithm tree") {

    val materializedAlgorithm = TemplateTreeHelper.materialize(template)

    val expectedAlgorithm =
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(LinearRegressionModel()),
          NodeTemplate(SparkGenericBagging(),
            Seq(
              LeafTemplate(Bayesian())
            )
          )
        )
      )

    assert(materializedAlgorithm == expectedAlgorithm)
  }

  /*test("Evaluation of metaalgorithm should use regressors of Ensembling nodes to calculate value from submembers'") {

    val metaAlgorithm =
      NodeTemplate(Bagging(),
        Seq(
          LeafTemplate(LinearRegressionModel()),
          NodeTemplate(MyStackingImpl(),
            Seq(
              LeafTemplate(DecisionTree()),
              LeafTemplate(DeepNeuralNetwork())
            )
          )
        )
      )

    val data: DataFrame = null
    metaAlgorithm.evaluateFitness(data, null, data) shouldBe 0.375 +- 0.01
  }*/


}


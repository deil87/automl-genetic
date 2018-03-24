package com.automl

import com.automl.helper.TemplateTreeHelper
import com.automl.template._
import com.automl.template.ensemble.bagging.Bagging
import com.automl.template.ensemble.stacking.GenericStacking
import com.automl.template.simple._
import org.apache.spark.sql.DataFrame
import org.scalatest.{FunSuite, Matchers}


class AlgorithmTreeSuite extends FunSuite with Matchers{


  val template =
    NodeTemplate(Bagging(),
      Seq(
        LeafTemplate(LinearRegressionModel()),
        NodeTemplate(Bagging(),
          Seq(
            LeafTemplate(Wildcard(List(Bayesian())))
          )
        )
      )
    )


  test("TemplateMemberHelper should materialize to algorithm tree") {

    val materializedAlgorithm = TemplateTreeHelper.materialize(template)

    val expectedAlgorithm =
      NodeTemplate(Bagging(),
        Seq(
          LeafTemplate(LinearRegressionModel()),
          NodeTemplate(Bagging(),
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


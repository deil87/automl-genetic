package com.automl

import com.automl.algorithm.{AlgorithmMemberFrom, LeafAlgorithm, NodeAlgorithm}
import com.automl.helper.TemplateTreeHelper
import com.automl.template._
import com.automl.template.ensemble.bagging.Bagging
import com.automl.template.simple._
import org.apache.spark.sql.DataFrame
import org.scalatest.{FunSuite, Matchers}


class AlgorithmTreeSuite extends FunSuite with Matchers{


  val template =
    NodeTemplate(Bayesian(),
      Seq(
        LeafTemplate(LinearRegressionModel()),
        NodeTemplate(RandomForest(),
          Seq(
            LeafTemplate(Wildcard(List(Bayesian())))
          )
        )
      )
    )


  test("TemplateMemberHelper should materialize to algorithm tree") {

    val materializedAlgorithm = TemplateTreeHelper.materialize(template)

    val expectedAlgorithm =
      NodeAlgorithm(AlgorithmMemberFrom(Bayesian()),
        Seq(
          LeafAlgorithm(AlgorithmMemberFrom(LinearRegressionModel())),
          NodeAlgorithm(AlgorithmMemberFrom(RandomForest()),
            Seq(
              LeafAlgorithm(AlgorithmMemberFrom(Bayesian()))
            )
          )
        )
      )

    assert(materializedAlgorithm == expectedAlgorithm)
  }

  test("Evaluation of metaalgorithm should use regressors of Ensembling nodes to calculate value from submembers'") {

    val metaAlgorithm =
      NodeAlgorithm(AlgorithmMemberFrom(Bagging()),
        Seq(
          LeafAlgorithm(AlgorithmMemberFrom(LinearRegressionModel())),
          NodeAlgorithm(AlgorithmMemberFrom(Bagging()),
            Seq(
              LeafAlgorithm(AlgorithmMemberFrom(DecisionTree())),
              LeafAlgorithm(AlgorithmMemberFrom(DeepNeuralNetwork()))
            )
          )
        )
      )

    val data: DataFrame = null
    metaAlgorithm.evaluate(data) shouldBe 0.375 +- 0.01
  }


}


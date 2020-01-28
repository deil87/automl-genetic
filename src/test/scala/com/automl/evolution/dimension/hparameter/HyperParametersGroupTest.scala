package com.automl.evolution.dimension.hparameter

import org.scalatest.concurrent.TimeLimitedTests
import org.scalatest.time.{Minutes, Span}
import org.scalatest.{FunSuite, Matchers}
import com.test.TestBase

class HyperParametersGroupTest extends FunSuite with TestBase{

  test("equals work properly") {
    LogisticRegressionHPGroup() == DecisionTreeHPGroup() shouldBe false

    val lrg1 = LogisticRegressionHPGroup(Seq(RegParamLR(Some(2)), ElasticNet(Some(3))))
    val lrg2 = LogisticRegressionHPGroup(Seq(ElasticNet(Some(3)), RegParamLR(Some(2))))
    val lrg3 = LogisticRegressionHPGroup(Seq(RegParamLR(Some(2)), ElasticNet(Some(4))))

    lrg1 == lrg2 shouldBe true
    lrg1 == lrg3 shouldBe false
  }
}

package com.automl.regressor

import org.scalatest.{FunSuite, Matchers}
import utils.test.TestBase

class MajorityVoteRegressorTest extends FunSuite with TestBase{

  test("majority regressor should return majority value") {
    val regressor = new MajorityVoteRegressor()
    val predictions = Seq("a", "b", "a", "b", "a")
    regressor.evaluate(predictions) should be("a")
  }

}

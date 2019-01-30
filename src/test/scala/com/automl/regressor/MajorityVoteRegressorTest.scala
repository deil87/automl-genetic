package com.automl.regressor

import org.scalatest.{FunSuite, Matchers}

class MajorityVoteRegressorTest extends FunSuite with Matchers{

  test("majority regressor should return majority value") {
    val regressor = new MajorityVoteRegressor()
    val predictions = Seq("a", "b", "a", "b", "a")
    regressor.evaluate(predictions) should be("a")
  }

}

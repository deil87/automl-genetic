package com.automl.teststrategy

trait TestStrategy {
  def trainingShare: Double
}

class TrainingTestSplitStrategy(val trainingShare: Double = 0.8) extends TestStrategy {

}


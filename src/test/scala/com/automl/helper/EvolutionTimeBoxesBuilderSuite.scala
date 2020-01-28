package com.automl.helper

import org.scalatest.{Matchers, WordSpec}
import utils.TestBase

class EvolutionTimeBoxesBuilderSuite extends WordSpec with TestBase{

  "EvolutionTimeBoxesBuilder" should {

    "generate equal ranges" in {
      val timeLimit = 1000
      val numEvolutions = 5
      val strategy: EqualEvolutionsStrategy = EqualEvolutionsStrategy(timeLimit, numEvolutions)
      val builder = EvolutionTimeBoxesBuilder(timeLimit, numEvolutions).withSplittingStrategy(strategy)
      val timeBoxes = builder.build
      timeBoxes should be(EvolutionTimeBoxes(Seq(TimeBox(0, 0, 200, 1000), TimeBox(1, 200, 400, 1000), TimeBox(2, 400 ,600, 1000), TimeBox(3, 600, 800, 1000), TimeBox(4, 800, 1000, 1000))))
    }
  }
}

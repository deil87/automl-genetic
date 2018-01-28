package com.automl.helper

import org.scalatest.{Matchers, WordSpec}

class EvolutionTimeBoxesBuilderSuite extends WordSpec with Matchers{

  "EvolutionTimeBoxesBuilder" should {

    "generate equal ranges" in {
      val timeLimit = 1000
      val numEvolutions = 5
      val strategy: EqualEvolutionsStrategy = EqualEvolutionsStrategy(timeLimit, numEvolutions)
      val builder = EvolutionTimeBoxesBuilder(timeLimit, numEvolutions).withSplittingStrategy(strategy)
      val timeBoxes = builder.build
      timeBoxes should be(EvolutionTimeBoxes(Seq(TimeBox(0, 200), TimeBox(1, 400), TimeBox(2 ,600), TimeBox(3 ,800), TimeBox(4, 1000))))
    }
  }
}

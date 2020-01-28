package com.automl.population

import com.automl.evolution.dimension.hparameter.BayesianHPGroup
import com.automl.template.simple.Bayesian
import org.scalatest.{FunSuite, Matchers}
import utils.test.TestBase

class HPPopulationTest extends FunSuite with TestBase{

  test("That `randomRelevantHPFieldFor` method will return relevant hp group") {
    val bayesian = Bayesian()
    val relevantHPGroup = HPPopulation.randomRelevantHPFieldFor(bayesian)

    relevantHPGroup.modelsHParameterGroups.size shouldBe 1
    relevantHPGroup.modelsHParameterGroups should contain theSameElementsAs( Seq(BayesianHPGroup()))
  }

}

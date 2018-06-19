package com.automl.evolution.mutation

import com.automl.template.ensemble.EnsemblingModelMember
import com.automl.template.ensemble.bagging.BaggingMember
import com.automl.template.simple.SimpleModelMember
import org.scalatest.{Matchers, WordSpec}

class MutationProbabilitiesSuite extends WordSpec with Matchers{

  "MutationProbabilities" should {

    "initialize with correct probabilities" in {

      val mp = MutationProbabilities(0.8, SimpleModelMember.poolOfSimpleModels.toSet, EnsemblingModelMember.poolOfEnsemblingModels)
      mp.probabilities(SimpleModelMember.DT).value shouldBe 0.26 +- 0.01
      mp.probabilities(BaggingMember.SimpleBagging).value shouldBe 0.1 +- 0.01
    }

    "update probabilities when we provide new knowledge about templateMember" in {

      val mp = MutationProbabilities(0.8, SimpleModelMember.poolOfSimpleModels.toSet, EnsemblingModelMember.poolOfEnsemblingModels)
      mp.probabilities(SimpleModelMember.DT).value shouldBe 0.26 +- 0.01

      mp.update(SimpleModelMember.DT, 42)

      mp.probabilities(SimpleModelMember.DT).value shouldBe 0.4 +- 0.01
      mp.noveltySection.get(SimpleModelMember.DT) shouldBe None

    }
  }
}

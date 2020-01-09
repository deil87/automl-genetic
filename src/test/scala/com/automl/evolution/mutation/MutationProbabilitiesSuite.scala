package com.automl.evolution.mutation

import com.automl.template.ensemble.EnsemblingModelMember
import com.automl.template.ensemble.bagging.BaggingMember
import com.automl.template.simple.SimpleModelMember
import org.scalatest.{Matchers, WordSpec}

class MutationProbabilitiesSuite extends WordSpec with Matchers{

  "MutationProbabilities" ignore {

    "initialize with correct probabilities" in {

      val mp = MutationProbabilities(0.8, SimpleModelMember.poolOfSimpleModels.toSet, EnsemblingModelMember.poolOfEnsemblingModels)
      mp.probabilities(SimpleModelMember.DT).value shouldBe 0.26 +- 0.01
      mp.probabilities(BaggingMember.SimpleBagging).value shouldBe 0.1 +- 0.01
    }

    "update probabilities when we provide new knowledge about templateMember" in {

      val mp = MutationProbabilities(0.8, SimpleModelMember.poolOfSimpleModels.toSet, EnsemblingModelMember.poolOfEnsemblingModels)
      mp.probabilities(SimpleModelMember.DT).value shouldBe 0.26 +- 0.01

      mp.moveToExperienceSection(SimpleModelMember.DT)
      mp.update(SimpleModelMember.DT, 42)

      mp.probabilities(SimpleModelMember.DT).value shouldBe 0.26 +- 0.01
      mp.noveltySection.get(SimpleModelMember.DT) shouldBe None
      mp.noveltySection.map(_._2.value).sum shouldBe 1 - 0.26 +- 0.01

    }

    "update probabilities when we provide new knowledge for multiple template members" in {

      val mp = MutationProbabilities(0.8, SimpleModelMember.poolOfSimpleModels.toSet, EnsemblingModelMember.poolOfEnsemblingModels)
      mp.probabilities(SimpleModelMember.DT).value shouldBe 0.26 +- 0.01

      mp.moveToExperienceSection(SimpleModelMember.DT)
      mp.update(SimpleModelMember.DT, 42)

      mp.probabilities(SimpleModelMember.DT).value shouldBe 0.26 +- 0.01
      mp.noveltySection.get(SimpleModelMember.DT) shouldBe None
      mp.noveltySection.map(_._2.value).sum shouldBe 1 - 0.26 +- 0.01

      //Add experience for next simple model
      mp.probabilities(SimpleModelMember.Bayesian).value shouldBe 0.26 +- 0.01

      mp.moveToExperienceSection(SimpleModelMember.Bayesian)
      mp.update(SimpleModelMember.Bayesian, 50)

      mp.noveltySection.get(SimpleModelMember.Bayesian) shouldBe None

      mp.probabilities(SimpleModelMember.DT).value shouldBe 0.0 +- 0.01
      mp.probabilities(SimpleModelMember.Bayesian).value shouldBe 0.53 +- 0.01


    }
  }
}

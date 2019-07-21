package com.automl.template

import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.dataset.Datasets
import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.problemtype.ProblemType
import com.automl.template.simple.{Bayesian, DecisionTree, RandomForest}
import org.scalatest.{FunSuite, Matchers}

class TemplateTreeTest extends FunSuite with Matchers {

  test("testHeight") {

  }

  test("mutation of some node is working") {

  }

  //TODO we can add option to mutate only on leaf level so that we can test with random template being transformed.
  test("that two templates with essentially the same base models but in different order are equal") {

    val individual: TemplateTree[TemplateMember] =
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(DecisionTree()),
          LeafTemplate(RandomForest()),
          LeafTemplate(Bayesian())
        )
      )

    val individual2: TemplateTree[TemplateMember] =
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(RandomForest()),
          LeafTemplate(DecisionTree()),
          LeafTemplate(Bayesian())
        )
      )

    println(individual.render)
    println(individual2.render)

    LeafTemplate(Bayesian()) == LeafTemplate(Bayesian()) shouldBe true
    LeafTemplate(Bayesian()) == LeafTemplate(DecisionTree()) shouldBe false

    individual shouldEqual individual2
  }

  // See bug https://github.com/scala-exercises/scala-exercises/issues/18
  test("Comparing Sets return expected results") {
    Set(1,2,3) sameElements Set(3,2,1) shouldBe false //  WTF?? expected to be true for Sets

    Set(1,2,3) diff Set(3,2,1) shouldBe Set()
  }

  test("that two templates with essentially the same base models but in different order are equal (case 2)") {

    val individual: TemplateTree[TemplateMember] =
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(DecisionTree()),
          LeafTemplate(RandomForest()),
          NodeTemplate(SparkGenericBagging(),
            Seq(
              LeafTemplate(DecisionTree()),
              LeafTemplate(RandomForest())
            )
          )
        )
      )

    val individual2: TemplateTree[TemplateMember] =
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(RandomForest()),
          NodeTemplate(SparkGenericBagging(),
            Seq(
              LeafTemplate(RandomForest()),
              LeafTemplate(DecisionTree())
            )
          ),
          LeafTemplate(DecisionTree())
        )
      )

    println(individual.render)
    println(individual2.render)

    individual shouldEqual individual2
  }

  test("that two templates should not be equal") {

    val individual: TemplateTree[TemplateMember] =
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(DecisionTree()),
          LeafTemplate(RandomForest()),
          NodeTemplate(SparkGenericBagging(),
            Seq(
              LeafTemplate(DecisionTree()),
              LeafTemplate(RandomForest())
            )
          )
        )
      )

    val individual2: TemplateTree[TemplateMember] =
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(RandomForest()),
          LeafTemplate(Bayesian()),
          NodeTemplate(SparkGenericBagging(),
            Seq(
              LeafTemplate(RandomForest()),
              LeafTemplate(DecisionTree())
            )
          ),
          LeafTemplate(DecisionTree())
        )
      )

    println(individual.render)
    println(individual2.render)

    individual == individual2 shouldBe false
  }

  test("that two templates should be considered equal") {

    val individual: TemplateTree[TemplateMember] =
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(DecisionTree()),
          NodeTemplate(SparkGenericBagging(),
            Seq(
              LeafTemplate(DecisionTree()),
              LeafTemplate(RandomForest()),
              LeafTemplate(Bayesian())
            )
          ),
          NodeTemplate(SparkGenericBagging(),
            Seq(
              LeafTemplate(DecisionTree()),
              LeafTemplate(Bayesian())
            )
          )
        )
      )

    val individual2: TemplateTree[TemplateMember] =
      NodeTemplate(SparkGenericBagging(),
        Seq(
          NodeTemplate(SparkGenericBagging(),
            Seq(
              LeafTemplate(Bayesian()),
              LeafTemplate(DecisionTree())
            )
          ),
          NodeTemplate(SparkGenericBagging(),
            Seq(
              LeafTemplate(RandomForest()),
              LeafTemplate(DecisionTree()),
              LeafTemplate(Bayesian())
            )
          ),
          LeafTemplate(DecisionTree())
        )
      )

    println(individual.render)
    println(individual2.render)

    individual == individual2 shouldBe true
  }

  test("evaluation does not cause error") {

    val individual: TemplateTree[TemplateMember] =
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(DecisionTree()),
          NodeTemplate(SparkGenericBagging(),
            Seq(
              LeafTemplate(DecisionTree()),
              LeafTemplate(Bayesian())
            )
          )
        )
      )

    val preparedGlassDF = Datasets.getGlassDataFrame(1234)
    val Array(trainingSplit, testSplit) = preparedGlassDF.randomSplit(Array(0.8, 0.2))

    val fitnessValue = individual.evaluateFitness(trainingSplit, testSplit, ProblemType.MultiClassClassificationProblem, hyperParamsMap = None)
  }

  //TODO we probably need to store only HPGroup that is suitable for a particular Member in the node
  test("every node of template tree gets its own HyperParameterField") {

    val individual: TemplateTree[TemplateMember] =
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(DecisionTree()),
          NodeTemplate(SparkGenericBagging(),
            Seq(
              LeafTemplate(DecisionTree()),
              LeafTemplate(Bayesian())
            )
          )
        )
      )

    val L0Bag: HyperParametersField = individual.internalHyperParamsMap.get
    L0Bag == L0Bag shouldBe true

    val L1DT: HyperParametersField = individual.subMembers(0).internalHyperParamsMap.get
    L0Bag shouldNot equal(L1DT)
  }
}

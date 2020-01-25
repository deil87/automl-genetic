package com.automl.template

import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.dataset.Datasets
import com.automl.evolution.dimension.hparameter._
import com.automl.evolution.mutation.HPMutationStrategy
import com.automl.problemtype.ProblemType
import com.automl.template.simple.{Bayesian, DecisionTree, LogisticRegressionModel, RandomForest}
import org.scalatest.{FunSuite, Matchers}

class TemplateTreeTest extends FunSuite with Matchers {

  test("testHeight") {

  }

  test("mutation of some node is working") {

  }

  //TODO we can add option to mutate only on leaf level so that we can test with random template being transformed.

  test("that two leaf templates are equal") {

    val dtTemplate = LeafTemplate(DecisionTree(DecisionTreeHPGroup(Seq(MaxDepth(Some(4.0))))))
    val dtTemplate2 = LeafTemplate(DecisionTree(DecisionTreeHPGroup(Seq(MaxDepth(Some(4.0))))))

    dtTemplate shouldEqual dtTemplate2
  }

  // See bug https://github.com/scala-exercises/scala-exercises/issues/18 ---
  test("Comparing Sets return expected results") {
    Set(1,2,3) sameElements Set(3,2,1) shouldBe false //  WTF?? expected to be true for Sets

    Set(1,2,3) diff Set(3,2,1) shouldBe Set()
  }

  test("that two templates with essentially the same base models but in different order are equal (case 2)") {

    val dtl = LeafTemplate(DecisionTree())
    val rfl = LeafTemplate(RandomForest())
    val rfl2 = LeafTemplate(RandomForest())
    val individual: TemplateTree[TemplateMember] =
      NodeTemplate(SparkGenericBagging(),
        Seq(
          dtl,
          rfl,
          NodeTemplate(SparkGenericBagging(),
            Seq(
              LeafTemplate(DecisionTree(DecisionTreeHPGroup(Seq(MaxDepth(Some(4.0)))))),
              rfl2
            )
          )
        )
      )

    val individual2: TemplateTree[TemplateMember] =
      NodeTemplate(SparkGenericBagging(),
        Seq(
          rfl,
          NodeTemplate(SparkGenericBagging(),
            Seq(
              rfl2,
              LeafTemplate(DecisionTree(DecisionTreeHPGroup(Seq(MaxDepth(Some(4.0))))))
            )
          ),
          dtl
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

  // TODO AG-203
  ignore("that two templates should be considered equal") {

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

  //TODO AG-203
  ignore("evaluation does not cause error") {

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

    val fitnessValue = individual.evaluateFitness(trainingSplit, testSplit, ProblemType.MultiClassClassificationProblem, hpFieldFromCoevolution = None)
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

  test("hyperparameters are being taken into account when we compare templates") {

    val individual: TemplateTree[TemplateMember] = LeafTemplate(DecisionTree())

    val individual2: TemplateTree[TemplateMember] = LeafTemplate(DecisionTree())
    individual2.internalHyperParamsMap = Some(HyperParametersField(
      Seq(
        DecisionTreeHPGroup(Seq(MaxDepth(Some(2.0))))
      )
    ))

    individual should equal(individual)
    individual shouldNot equal(individual2)
  }

  test("hyperparameters that are passed through constructor ovverride everything else") {

    val sameLogisticRegressionHPGroup = LogisticRegressionHPGroup(Seq(RegParamLR(Some(0.2)), ElasticNet(Some(0.2))))

    val individual: TemplateTree[TemplateMember] = LeafTemplate(LogisticRegressionModel(sameLogisticRegressionHPGroup))
    val individual2: TemplateTree[TemplateMember] = LeafTemplate(LogisticRegressionModel(sameLogisticRegressionHPGroup))

    println(individual)

    individual should equal(individual2)

    val anotherLogisticRegressionHPGroup = LogisticRegressionHPGroup(Seq(RegParamLR(Some(0.4)), ElasticNet(Some(0.5))))


    val individual3: TemplateTree[TemplateMember] = LeafTemplate(LogisticRegressionModel(anotherLogisticRegressionHPGroup))

    individual should not equal(individual3)
  }
}

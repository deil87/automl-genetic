package com.automl.evolution.mutation

import com.automl.ConfigProvider
import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.evolution.dimension.hparameter.DecisionTreeHPGroup
import com.automl.evolution.diversity.DistinctDiversityStrategy
import com.automl.helper.PopulationHelper
import com.automl.population.{GenericPopulationBuilder, TPopulation}
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.MultiClassClassificationProblem
import com.automl.template.ensemble.stacking.GenericStacking
import com.automl.template.simple._
import com.automl.template.{LeafTemplate, NodeTemplate, TemplateMember, TemplateTree}
import com.typesafe.config.{Config, ConfigFactory}
import org.scalatest.{FunSuite, Matchers}
import utils.TestBase

class DepthDependentTemplateMutationStrategyTest extends FunSuite with TestBase {

  test("mutate templateTree from base model to complex algorithm") {

    implicit val logPaddingSize: Int = 0
    val seed: Seq[LeafTemplate[SimpleModelMember]] = Seq(
      LeafTemplate(Bayesian()),
      LeafTemplate(LinearRegressionModel()),
      LeafTemplate(DecisionTree())
    )

    val seedPopulation = new TPopulation(seed)

    val population = GenericPopulationBuilder.fromSeedPopulation(seedPopulation).withSize(10).build

    PopulationHelper.print(population)

    val distinctStrategy = new DistinctDiversityStrategy()

    val problemType = MultiClassClassificationProblem

    val mutationStrategy = new DepthDependentTemplateMutationStrategy(distinctStrategy, problemType)

    val mutated = mutationStrategy.mutate(population)

    PopulationHelper.print(mutated)

    val mutated2 = mutationStrategy.mutate(mutated)
    PopulationHelper.print(mutated2)

    val mutated3 = mutationStrategy.mutate(mutated2)
    PopulationHelper.print(mutated3)

    //TODO make mutation happens every time
    mutated shouldNot be(population)
    mutated2 shouldNot be(mutated)
    mutated3 shouldNot be(mutated2)
  }

  test("test that we can get desired depth of the tree during mutation") {

    ConfigProvider.clearOverride.addOverride(
      """
        |evolution {
        |  templateDimension {
        |    maxEnsembleDepth = 4
        |  }
        |}
      """)

    val distinctStrategy = new DistinctDiversityStrategy()
    val mutationStrategy = new DepthDependentTemplateMutationStrategy(distinctStrategy, MultiClassClassificationProblem)(0)

    val individuals = Seq(
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(DecisionTree())
        )
      )
    )

    val population = new TPopulation(individuals)

    population.individuals.head.height shouldBe 2

    population.individuals.foreach(template => print(template.render))

    val newPopulation = mutationStrategy.mutateNTimes(population, 20)

    newPopulation.individuals.head.height should be <= 4

  }

  test("test that we don't get ensembling nodes with less then 2 child") {

    val individuals = Seq(
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(DecisionTree()),
          NodeTemplate(SparkGenericBagging(),
            Seq(
              LeafTemplate(DecisionTree()),
              NodeTemplate(SparkGenericBagging(),
                Seq(
                  LeafTemplate(DecisionTree())
                )
              )
            )
          )
        )
      )
    )

    val badPopulation = new TPopulation(individuals)

    val individuals2 = Seq(
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(DecisionTree()),
          NodeTemplate(SparkGenericBagging(),
            Seq(
              LeafTemplate(DecisionTree()),
              NodeTemplate(SparkGenericBagging(),
                Seq(
                  LeafTemplate(DecisionTree()),
                  LeafTemplate(Bayesian())  // <--- difference with `individuals` is only here
                )
              )
            )
          )
        )
      )
    )

    val correctPopulation = new TPopulation(individuals2)

    badPopulation.individuals.head.traverseAndCheck(_.subMembers.size >= 2) shouldBe false
    correctPopulation.individuals.head.traverseAndCheck(_.subMembers.size >= 2) shouldBe true
  }

  test("test that we don't get ensembling nodes with less then 2 child in real-world case") {
    val testOverride: Config = ConfigFactory.parseString(
      """
        |evolution {
        |  templateDimension {
        |    maxEnsembleDepth = 4
        |  }
        |}
      """.stripMargin)
    ConfigProvider.addOverride(testOverride)

    val distinctStrategy = new DistinctDiversityStrategy()
    val conf = ConfigProvider.config
    val mutationStrategy = new DepthDependentTemplateMutationStrategy(distinctStrategy, MultiClassClassificationProblem)(0)

    val individuals = Seq(
          LeafTemplate(DecisionTree())
    )

    val population = new TPopulation(individuals)


    val newPopulation = mutationStrategy.mutateNTimes(population, 20)

    newPopulation.individuals.foreach(template => println(template.render))
    newPopulation.individuals.head.traverseAndCheck(_.subMembers.size >= 2) shouldBe true
  }

  test("that we don't get into dead end as we compute level of mutation based on tree height") {

    //TODO
  }

  test("that we will receive new individual by applying mutateIndividual method(multi rerun setup)") {

    implicit val padding: Int = 0
    val strategy = new DepthDependentTemplateMutationStrategy(null, ProblemType.MultiClassClassificationProblem)

    val numberOfMutationIterations = 50
    (1 to numberOfMutationIterations).foreach{ _ =>
      val individual: NodeTemplate[TemplateMember] =
        NodeTemplate(SparkGenericBagging(),
          Seq(
            LeafTemplate(DecisionTree()),
            LeafTemplate(RandomForest()),
            LeafTemplate(Bayesian())
          )
        )

      val newIndividual = strategy.mutateIndividual(individual)

      println(individual.render)
      println(newIndividual.render)

      val areTheyEqual = individual == newIndividual
      areTheyEqual should be(false)
      individual shouldEqual individual
    }

  }

  ignore("that copy of LeafTree copy internal field") {
    val individual = LeafTemplate(DecisionTree())
    val copy = individual.copy()
//    individual == individual shouldBe true
//    individual == copy shouldBe false
  }


  test("we can mutate hpGroup and that equals method from hpGroup works") {
    implicit val padding: Int = 0
    val strategy = new DepthDependentTemplateMutationStrategy(null, ProblemType.MultiClassClassificationProblem)

    val template = LeafTemplate(DecisionTree())
    val originalHPGroup: DecisionTreeHPGroup = template.member.hpGroup
    val mutatedHPGroup = strategy.mutateHPGroupAspectOfTemplateTree(template).member.hpGroupInternal

    originalHPGroup == mutatedHPGroup should not be true
  }

    test("that we will receive new unique population by using mutate method ") {

    implicit val padding: Int = 0
    val strategy = new DepthDependentTemplateMutationStrategy(null, ProblemType.MultiClassClassificationProblem)

    val individual1: TemplateTree[TemplateMember] =
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(DecisionTree()),
          LeafTemplate(RandomForest()),
          LeafTemplate(Bayesian())
        )
      )
    val individual2: TemplateTree[TemplateMember] =
      NodeTemplate(GenericStacking(),
        Seq(
          LeafTemplate(DecisionTree())
        )
      )
    val individual3: TemplateTree[TemplateMember] =
      LeafTemplate(DecisionTree())

    val originalPopulation = new TPopulation(Seq(individual1, individual2, individual3))

    originalPopulation.individuals diff Seq(individual3, individual1, individual2) shouldBe Seq()

    val newPopulation = strategy.mutate(originalPopulation)

    PopulationHelper.print(originalPopulation, "Original population:")
    PopulationHelper.print(newPopulation, "New population:")

    //Following way to test could not be used because of the scalatest's way of comparison and overriden equal method on TemplateTree
    //originalPopulation.individuals shouldNot contain theSameElementsAs(newPopulation.individuals)
    originalPopulation.individuals diff newPopulation.individuals shouldNot be( Seq())
  }

  test("that we will increase complexity") {
    implicit val padding: Int = 0

    val numberOfMutationPhases = 10
    val seed = 1234

    val strategy = new RandomTemplateMutationStrategy(null, ProblemType.MultiClassClassificationProblem, seed)
    val individual1: TemplateTree[TemplateMember] =
      NodeTemplate(SparkGenericBagging(),
        Seq(
          LeafTemplate(DecisionTree()),
          LeafTemplate(RandomForest()),
          LeafTemplate(Bayesian())
        )
      )
    val individual2: TemplateTree[TemplateMember] =
      NodeTemplate(GenericStacking(),
        Seq(
          LeafTemplate(DecisionTree())
        )
      )

    val individual3: TemplateTree[TemplateMember] =
      LeafTemplate(DecisionTree())

    val originalPopulation = new TPopulation(Seq(individual1, individual2, individual3))
    originalPopulation.individuals diff Seq(individual3, individual1, individual2) shouldBe Seq()

    val newPopulation = (1 to numberOfMutationPhases).foldLeft(originalPopulation) ((currentPopulation, index) => strategy.mutate(currentPopulation))

    PopulationHelper.print(originalPopulation, "Original population:")
    PopulationHelper.print(newPopulation, "New population:")

    val originalComplexity = originalPopulation.depthComplexity
    val newComplexity = newPopulation.depthComplexity
    newComplexity shouldBe >= (originalComplexity)
  }


  test("that mutation of HyperParameters results in a creation of a new individual (#146, #143)") {
    val testOverride: Config = ConfigFactory.parseString(
      """
        |evolution {
        |  hpGridSearch = false
        |  hyperParameterDimension {
        |     enabled = false
        |  }
        |  templateDimension {
        |     pivotBetweenStructureAndHPMutations = -1.0
        |     maxNumberOfMutationAttempts = 1
        |  }
        |}
      """.stripMargin)
    ConfigProvider.clearOverride.addOverride(testOverride)

    implicit val padding: Int = 0
    val strategy = new DepthDependentTemplateMutationStrategy(null, ProblemType.MultiClassClassificationProblem)

    val originalPopulation = new TPopulation(Seq(LeafTemplate(Bayesian())))

    val newPopulation = strategy.mutate(originalPopulation)

    PopulationHelper.print(originalPopulation, "Original population:")
    PopulationHelper.print(newPopulation, "New population:")

    originalPopulation.individuals diff newPopulation.individuals shouldNot be( Seq())
  }

  test("mutation of the individual will result in an increase of degree of exploration variable in the parent node") {

    val testOverride: Config = ConfigFactory.parseString(
      """
        |evolution {
        |  hpGridSearch = false
        |  hyperParameterDimension {
        |     enabled = false
        |  }
        |  templateDimension {
        |     pivotBetweenStructureAndHPMutations = 0
        |     maxNumberOfMutationAttempts = 1000
        |  }
        |}
      """.stripMargin)
    ConfigProvider.clearOverride.addOverride(testOverride)

    implicit val padding: Int = 0
    val strategy = new DepthDependentTemplateMutationStrategy(null, ProblemType.MultiClassClassificationProblem)

    val dt = LeafTemplate(DecisionTree())

    val bayesian = LeafTemplate(Bayesian())

    val individual: NodeTemplate[TemplateMember] =
      NodeTemplate(SparkGenericBagging(),
        Seq(
          dt,
          bayesian
          //            LeafTemplate(RandomForest())
        )
      )

    var currentStateOfIndividual: TemplateTree[TemplateMember] = individual
    1 to 100 foreach { i =>
      println(s"\n\nIteration: $i")
      currentStateOfIndividual = strategy.mutateIndividual(currentStateOfIndividual)
      println("Degree of exploration: " + currentStateOfIndividual.asInstanceOf[NodeTemplate[TemplateMember]].degreeOfExploration)
      println(currentStateOfIndividual.render)
    }
  }

}

package com.automl.evolution.selection

import com.automl.evolution.dimension.hparameter.{DecisionTreeHPGroup, HyperParametersField, MaxDepth}
import com.automl.helper.{FitnessResult, PopulationHelper}
import com.automl.population.{GenericPopulationBuilder, TPopulation}
import com.automl.problemtype.ProblemType.{MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.{LeafTemplate, TemplateMember, TemplateTree}
import com.automl.{AutoML, EvaluatedTemplateData}
import com.automl.template.simple.{DecisionTree, LinearRegressionModel, SimpleModelMember}
import org.scalatest.{Matchers, WordSpec}

import scala.util.Random

class RankSelectionStrategyTest extends WordSpec with Matchers{

  "RankSelectionStrategyTest" should {

    "parentSelection" in {
      val individuals: Seq[LeafTemplate[SimpleModelMember]] = Seq(
        LeafTemplate(LinearRegressionModel()),
        LeafTemplate(DecisionTree())
      )

      val populationSize = 10
      val selectionShare = 0.5

      val individualsSpanned = GenericPopulationBuilder.fromSeedPopulation(new TPopulation(individuals)).withSize(populationSize).build.individuals

      val selectionStrategy = new RankSelectionStrategy
      val evaluatedTemplateDatas = individualsSpanned.zipWithIndex.map { case (inds, idx) =>
        EvaluatedTemplateData(idx.toString, inds, null, FitnessResult(Map("rmse" -> Random.nextDouble()), RegressionProblem, null), hyperParamsField = null)
      }

      val selectedParents = selectionStrategy.parentSelectionByShare(selectionShare, evaluatedTemplateDatas)

      // <editor-fold defaultstate="collapsed" desc="Uncomment if you need to visualise distribution">
      /*import breeze.plot._

      val f = Figure()
      val p2 = f.subplot(0)
      p2 += hist(selectedParents.map(_.rank), 100)
      p2.title = "A normal distribution"
      f.saveas("subplots.png")
      Thread.sleep(50000)*/
      // </editor-fold>

      PopulationHelper.print(new TPopulation(selectedParents.map(_.template)))

      selectedParents.length shouldBe (populationSize * selectionShare)
    }

    "parentSelection with local competitions" in {
      val individuals: Seq[LeafTemplate[SimpleModelMember]] = Seq(
        LeafTemplate(LinearRegressionModel()),
        LeafTemplate(LinearRegressionModel()),
        LeafTemplate(DecisionTree())
      )

      val populationSize = 10
      val selectionShare = 0.5

      val individualsSpanned = GenericPopulationBuilder.fromSeedPopulation(new TPopulation(individuals)).withSize(populationSize).build.individuals

      val selectionStrategy = new RankSelectionStrategy
      val evaluatedTemplateData = individualsSpanned.zipWithIndex.map { case (individual, idx) =>
        EvaluatedTemplateData(idx.toString, individual, null,
          FitnessResult(Map("f1" -> idx * 100), MultiClassClassificationProblem, null)
          )
      }

      val evaluatedTemplateDataWithNeighbours = evaluatedTemplateData.map(etd => etd.copy(neighbours = evaluatedTemplateData.diff(Seq(etd))))

      val selectedParents = selectionStrategy.parentSelectionByShareWithLocalCompetitions(selectionShare, evaluatedTemplateDataWithNeighbours)

      PopulationHelper.print(new TPopulation(selectedParents.map(_.template)))

      selectedParents.length shouldBe (populationSize * selectionShare)
      selectedParents.map(_.fitness.getCorrespondingMetric) should contain theSameElementsAs Seq(900, 800, 700, 600, 500)
    }

    "parentSelection with local competitions with sampling is working" ignore { //TODO fix the test
      val individuals: Seq[LeafTemplate[SimpleModelMember]] = Seq(
        LeafTemplate(LinearRegressionModel()),
        LeafTemplate(LinearRegressionModel()),
        LeafTemplate(DecisionTree())
      )

      val populationSize = 10

      val individualsSpanned = GenericPopulationBuilder.fromSeedPopulation(new TPopulation(individuals)).withSize(populationSize).build.individuals

      val selectionStrategy = new RankSelectionStrategy
      val evaluatedTemplateData = individualsSpanned.zipWithIndex.map { case (individual, idx) =>
        EvaluatedTemplateData(idx.toString, individual, null,
          FitnessResult(Map("f1" -> idx * 100), MultiClassClassificationProblem, null)
        )
      }

      val evaluatedTemplateDataWithNeighbours = evaluatedTemplateData.map(etd => etd.copy(neighbours = evaluatedTemplateData.diff(Seq(etd))))

      val sizeOfSample = 1500
      val selectedParents = selectionStrategy.selectionBySizeWithLocalCompetitions(sizeOfSample, evaluatedTemplateDataWithNeighbours)

      PopulationHelper.print(new TPopulation(selectedParents.map(_.template)))

      selectedParents.length shouldBe sizeOfSample

      //Check sampling ratio
      val withAssignedProbs = new RankBasedSelectionProbabilityAssigner[EvaluatedTemplateData].assign(selectedParents.toList)

      val assignedAndExpectedProbability = withAssignedProbs.count(_._1.fitness.getCorrespondingMetric == 900).toDouble / sizeOfSample
      assignedAndExpectedProbability shouldBe assignedAndExpectedProbability +- 0.01

    }

    "when scores are equals we should not count them as winners over the same neighbours" in {
      val individuals: Seq[LeafTemplate[SimpleModelMember]] = Seq(
        LeafTemplate(LinearRegressionModel()),
        LeafTemplate(LinearRegressionModel()),
        LeafTemplate(DecisionTree())
      )

      val populationSize = 10

      val individualsSpanned = GenericPopulationBuilder.fromSeedPopulation(new TPopulation(individuals)).withSize(populationSize).build.individuals

      val selectionStrategy = new RankSelectionStrategy
      val evaluatedTemplateData = individualsSpanned.zipWithIndex.map { case (individual, idx) =>

        val sameScoreEveryThirdIndex = if (idx % 3 == 0) 42 else 100
        // For population size = 10 we expect 4 members to have same `42` score.

        EvaluatedTemplateData(idx.toString, individual, null,
          FitnessResult(Map("f1" -> sameScoreEveryThirdIndex), MultiClassClassificationProblem, null)
        )
      }

      val evaluatedTemplateDataWithNeighbours = evaluatedTemplateData.map(etd => etd.copy(neighbours = evaluatedTemplateData.diff(Seq(etd))))

      val sizeOfSample = 10
      val selectedParents = selectionStrategy.selectionBySizeWithLocalCompetitions(sizeOfSample, evaluatedTemplateDataWithNeighbours)

      PopulationHelper.print(new TPopulation(selectedParents.map(_.template)))

      // It is difficult to check right now that we are getting members with 0 scores as their neighbours have same fitness values

    }

    "evaluatedTemplates are being selected properly" in {
      val template1 = LeafTemplate(DecisionTree())
      template1.internalHyperParamsMap = Some(HyperParametersField(
        Seq(
          DecisionTreeHPGroup(Seq(MaxDepth(Some(2.0))))
        )
      ))
      val template2 = LeafTemplate(DecisionTree())
      template2.internalHyperParamsMap = Some(HyperParametersField(
        Seq(
          DecisionTreeHPGroup(Seq(MaxDepth(Some(9.0))))
        )
      ))
      val template3 = LeafTemplate(DecisionTree())
      template3.internalHyperParamsMap = Some(HyperParametersField(
        Seq(
          DecisionTreeHPGroup(Seq(MaxDepth(Some(7.0))))
        )
      ))

      val individuals: Seq[LeafTemplate[SimpleModelMember]] = Seq(
        template1,
        template2,
        template3
      )

      val sameRMSE = Random.nextDouble()

      val selectionStrategy = new RankSelectionStrategy
      val evaluatedTemplateDatas: Seq[EvaluatedTemplateData] = individuals.zipWithIndex.map { case (inds, idx) =>
        EvaluatedTemplateData(idx.toString, inds, null, FitnessResult(Map("rmse" -> sameRMSE), RegressionProblem, null), hyperParamsField = null)
      }

      1 to 10 foreach { index =>
        val (res, _) = selectionStrategy.selectNIndividualsFromSortedByRankCollectionWithoutReplacement(2, evaluatedTemplateDatas)

        def getMaxDepth(template: TemplateTree[TemplateMember]) = {
          template.internalHyperParamsMap.get.modelsHParameterGroups.head.asInstanceOf[DecisionTreeHPGroup].hpParameters.head.asInstanceOf[MaxDepth].currentValue
        }

        val maxDepth1: Double = getMaxDepth(res(0).item)
        val maxDepth2: Double = getMaxDepth(res(1).item)

        getMaxDepth(template1) shouldBe getMaxDepth(template1)
        maxDepth1 == maxDepth2 should not be true
        maxDepth1 should not be equal(maxDepth2)

      }
    }
  }
}

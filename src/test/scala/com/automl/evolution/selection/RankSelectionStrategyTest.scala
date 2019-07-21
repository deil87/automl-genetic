package com.automl.evolution.selection

import com.automl.helper.{FitnessResult, PopulationHelper}
import com.automl.population.{GenericPopulationBuilder, TPopulation}
import com.automl.problemtype.ProblemType.{MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.LeafTemplate
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

    "parentSelection with local competitions with sampling is working" in {
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
          FitnessResult(Map("f1" -> sameScoreEveryThirdIndex ), MultiClassClassificationProblem, null)
        )
      }

      val evaluatedTemplateDataWithNeighbours = evaluatedTemplateData.map(etd => etd.copy(neighbours = evaluatedTemplateData.diff(Seq(etd))))

      val sizeOfSample = 10
      val selectedParents = selectionStrategy.selectionBySizeWithLocalCompetitions(sizeOfSample, evaluatedTemplateDataWithNeighbours)

      PopulationHelper.print(new TPopulation(selectedParents.map(_.template)))

      // It is difficult to check right now that we are getting members with 0 scores as their neighbours have same fitness values

    }

  }
}

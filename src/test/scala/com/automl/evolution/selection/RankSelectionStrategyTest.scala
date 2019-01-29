package com.automl.evolution.selection

import com.automl.helper.{FitnessResult, PopulationHelper}
import com.automl.problemtype.ProblemType.RegressionProblem
import com.automl.template.LeafTemplate
import com.automl.{AutoML, EvaluatedTemplateData, TPopulation}
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

      val individualsSpanned = TPopulation.fromSeedPopulation(new TPopulation(individuals)).withSize(populationSize).build.individuals

      val selectionStrategy = new RankSelectionStrategy
      val evaluatedTemplateDatas = individualsSpanned.zipWithIndex.map { case (inds, idx) =>
        EvaluatedTemplateData(idx.toString, inds, null, FitnessResult(Map("rmse" -> Random.nextDouble()), RegressionProblem, null))
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

  }
}

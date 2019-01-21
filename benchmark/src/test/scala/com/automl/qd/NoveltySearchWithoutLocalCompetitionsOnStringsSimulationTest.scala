package com.automl.qd

import org.apache.commons.lang3.StringUtils
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpec}

import scala.util.Random

class NoveltySearchWithoutLocalCompetitionsOnStringsSimulationTest extends WordSpec with Matchers with BeforeAndAfterAll  {


    "QD search" should {


      "return 1000 as a score for `900000`" in {

        val inputString = "900000"
        val score = new NoveltySearchWithoutNichesOnStringsSimulation().evaluateIndividual(inputString)

        score should be(1000)

      }

      "return 1000 as a score for `000009`" in {

        val inputString = "000009"
        val score = new NoveltySearchWithoutNichesOnStringsSimulation().evaluateIndividual(inputString)

        score should be(1000)

      }

      "return 0 as a score for `900009`" in {

        val inputString = "900009"
        val score = new NoveltySearchWithoutNichesOnStringsSimulation().evaluateIndividual(inputString)

        score should be(0)

      }

      "return 3 as a score" in {

        val inputString = "300000"
        val score = new NoveltySearchWithoutNichesOnStringsSimulation().evaluateIndividual(inputString)

        score should be(3)

      }

      "return evaluated seq " in {

        val inputString = Seq("300000", "900000", "123004")
        val scores = new NoveltySearchWithoutNichesOnStringsSimulation().evaluatePopulation(inputString)

        scores should contain theSameElementsAs Seq(EvaluatedStr("300000", 3), EvaluatedStr("900000", 1000), EvaluatedStr("123004", 6))

      }

      "mutate individual randomly " in {

        val inputString = "300000"
        val mutated = new NoveltySearchWithoutNichesOnStringsSimulation().mutateIndividual(inputString)

        val int = Integer.parseInt(mutated)
        int <= 399999 should be(true)

      }

      "sort desc" in {

        val sorted = Seq(EvaluatedStr("", 4), EvaluatedStr("", 1), EvaluatedStr("", 7)).sortWith(_.score > _.score)
        sorted.map(_.score).head should be(7)

      }

      "has correct levenstein distance algo" in {

        val book = "book"
        val back = "back"
        val distance = StringUtils.getLevenshteinDistance(book, back)
        distance should be(2)

      }


      "determineMorphologicalNicheBasedOnMorphology" in {

        val inputString = "800000"
        val nicheIndex = new NoveltySearchWithoutNichesOnStringsSimulation().determineMorphologicalNicheBasedOnMorphology(inputString)
        nicheIndex should be(0)

        val inputString2 = "300064"
        val nicheIndex2 = new NoveltySearchWithoutNichesOnStringsSimulation().determineMorphologicalNicheBasedOnMorphology(inputString2)
        nicheIndex2 should be(1)

      }

      "find all local maximums" in {

        val population = (1 to 500).map{ case index => f"${Random.nextInt(999999)}%06d"}

        val simulation = new NoveltySearchWithoutNichesOnStringsSimulation()
        val evolved = simulation.run(population, 100)

        val winners = simulation.evaluatePopulation(evolved).sortWith(_.score > _.score).take(5)

        println(winners.mkString(","))
        true should be(true)

      }

    }

}


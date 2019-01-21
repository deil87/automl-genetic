package com.automl.qd

import org.apache.commons.lang3.StringUtils
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpec}

import scala.util.Random

class NoveltySearchWithLocalCompetitionsOnStringsSimulationTest extends WordSpec with Matchers with BeforeAndAfterAll  {


    "QD search" should {


      "return 1000 as a score for `900000`" in {

        val inputString = "900000"
        val score = new NoveltySearchWithLocalCompetitionsOnStringsSimulation().evaluateIndividual(inputString)

        score should be(1000)

      }

      "return 1000 as a score for `000009`" in {

        val inputString = "000009"
        val score = new NoveltySearchWithLocalCompetitionsOnStringsSimulation().evaluateIndividual(inputString)

        score should be(1000)

      }

      "return 0 as a score for `900009`" in {

        val inputString = "900009"
        val score = new NoveltySearchWithLocalCompetitionsOnStringsSimulation().evaluateIndividual(inputString)

        score should be(0)

      }

      "return 3 as a score" in {

        val inputString = "300000"
        val score = new NoveltySearchWithLocalCompetitionsOnStringsSimulation().evaluateIndividual(inputString)

        score should be(3)

      }

      "return evaluated seq " in {

        val inputString = Seq("300000", "900000", "123004")
        val scores = new NoveltySearchWithLocalCompetitionsOnStringsSimulation().evaluatePopulation(inputString)

        scores should contain theSameElementsAs Seq(EvaluatedStr("300000", 3), EvaluatedStr("900000", 1000), EvaluatedStr("123004", 6))

      }

      "mutate individual randomly " in {

        val inputString = "300000"
        val mutated = new NoveltySearchWithLocalCompetitionsOnStringsSimulation().mutateIndividual(inputString)

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
        val nicheIndex = new NoveltySearchWithLocalCompetitionsOnStringsSimulation().determineMNicheIndexBasedOnMorphologicalCriteria(inputString)
        nicheIndex should be(0)

        val inputString2 = "300064"
        val nicheIndex2 = new NoveltySearchWithLocalCompetitionsOnStringsSimulation().determineMNicheIndexBasedOnMorphologicalCriteria(inputString2)
        nicheIndex2 should be(1)

      }

      "compute correct local performances" in {

        val ind1 = EvaluatedStr("a", 350)
        val ind2 = EvaluatedStr("b", 500)
        val ind3 = EvaluatedStr("c", 400)
        val ind4 = EvaluatedStr("d", 100)
        val ind5 = EvaluatedStr("e", 900)
        val ind6 = EvaluatedStr("f", 100)
        val input = Seq(
          EvaluatedStrWithNeighbours(ind1, Seq(ind2, ind3)),
          EvaluatedStrWithNeighbours(ind2, Seq(ind1, ind3)),
          EvaluatedStrWithNeighbours(ind3, Seq(ind1, ind2)),
          EvaluatedStrWithNeighbours(ind4, Seq(ind3, ind5)),
          EvaluatedStrWithNeighbours(ind5, Seq(ind4, ind6)),
          EvaluatedStrWithNeighbours(ind6, Seq(ind4, ind5))
        )
        val localPerformances = new NoveltySearchWithLocalCompetitionsOnStringsSimulation(2, 100).evaluateLocalPerformances(input)

        println(localPerformances.map(ind => ind.individual + "_" + ind.localScore).mkString("\n"))

//        Expect following result:
//
//        EvaluatedStr(a,350)_0
//        EvaluatedStr(b,500)_200  <- max
//        EvaluatedStr(c,400)_100
//        EvaluatedStr(d,100)_0
//        EvaluatedStr(e,900)_200 <- max
//        EvaluatedStr(f,100)_0
      }

      "find all local maximums since we value diversity of our solutions" in {

        val population = (1 to 500).map{ case index => f"${Random.nextInt(999999)}%06d"}

        val simulation = new NoveltySearchWithLocalCompetitionsOnStringsSimulation()
        val evolved = simulation.run(population, 200)

        val winners = simulation.evaluateAndGetNichesPerformances(evolved).sortWith(_.localScore > _.localScore).take(10)

        println(winners.mkString(","))
        true should be(true)

      }

    }

}


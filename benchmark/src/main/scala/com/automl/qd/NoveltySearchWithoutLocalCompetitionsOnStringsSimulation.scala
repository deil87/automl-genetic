package com.automl.qd

import scala.util.Random

// one of the version of the Quality Diversity search
class NoveltySearchWithoutNichesOnStringsSimulation {


  def evaluateIndividual(individual: String): Int = {
    var sum = 0
    if(individual.substring(0, 1).toInt >= 7 && individual.map(_.toString.toInt).tail.forall(_ <= 3)) {
      sum += 500
    }
    if(individual.substring(individual.length-1).toInt >= 7 && individual.map(_.toString.toInt).take(individual.length-1).forall(_ <= 3)) {
      sum += 1000
    }
    sum += individual.map(_.toString.toInt).filter( _ <=3).sum
    sum
  }

  def evaluatePopulation(population: Seq[String]): Seq[EvaluatedStr] = {
    population.map(str => EvaluatedStr(str, evaluateIndividual(str)))
  }

  def distance() = {
    //    StringUtils.getLevenshteinDistance()
  }

  //Returns niche for particular individual
  //Niches are defined manually by if/else logic
  def determineMorphologicalNicheBasedOnMorphology(individual: String):Int = {

    //has number that is higher that 6 in left half
    if(individual.substring(0, 2).map(_.toString.toInt).exists(_ > 6)) {
      0
    }
    else {
      1
    }

  }

  def run(population: Seq[String], numberOfGenerations: Int): Seq[String] = {

    var evolvedPopulation = population
    for( i <- 0 until numberOfGenerations) {
      val evolved: Seq[String] = runSingleGeneration(evolvedPopulation)
      evolvedPopulation = evolved
    }
    evolvedPopulation
  }

  private def runSingleGeneration(population: Seq[String]): Seq[String] = {

    val evaluatedPopulation = evaluatePopulation(population)

    val selectedParents = evaluatedPopulation.sortWith(_.score > _.score).take(evaluatedPopulation.length / 2)

    val offspring = selectedParents.map(parent => mutateIndividual(parent.individual))

    val populationWithOffspring = evaluatedPopulation ++ evaluatePopulation(offspring)

    val newGenaration = populationWithOffspring.sortWith(_.score > _.score).take(population.length)
    println(newGenaration.mkString(","))

    newGenaration.map(_.individual)
  }

  def mutateIndividual(individual: String):String = {
    val randomInt = Random.nextInt(10)
    val randomIndex = Random.nextInt(individual.length)
    individual.map(_.toString).updated(randomIndex, randomInt.toString).mkString
  }
}

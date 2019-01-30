package com.automl.qd

import org.apache.commons.lang3.StringUtils

import scala.collection.mutable
import scala.util.Random

// one of the version of the Quality Diversity search
/**
  *
  * @param neighbourhoodSize size of neigbourhood excluding element itself
  * @param localPerformanceCoefficient
  */
class NoveltySearchWithLocalCompetitionsOnStringsSimulation(val neighbourhoodSize:Int  = 10, val localPerformanceCoefficient:Int = 100) {


  // Evaluation should be the same for everione because it should be based on the same criteria universally. We will use niches later when doing selection based on local performances
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

  //After evaluation we will be able find  performance within neighbourhood based on distance notion
  def evaluatePopulation(population: Seq[String]): Seq[EvaluatedStr] = {
    population.map(str => EvaluatedStr(str, evaluateIndividual(str)))
  }

  def evaluateLocalPerformances(individuals: Seq[EvaluatedStrWithNeighbours]): Seq[EvaluatedStrWithLocalScores] = {
    individuals.map{ indWithNeighbrs =>
      EvaluatedStrWithLocalScores(
        indWithNeighbrs.individual,
        indWithNeighbrs.neighbours.count(_.score < indWithNeighbrs.individual.score) * localPerformanceCoefficient
      )
    }
  }

  def distance(from: EvaluatedStr, distanceTo: EvaluatedStr): DistanceToNeighbour = {
    val levenshteinDist = StringUtils.getLevenshteinDistance(from.individual, distanceTo.individual)
    DistanceToNeighbour(distanceTo, levenshteinDist)
  }

  //TODO unused
  //Returns niche for particular individual
  //Niches are defined manually by if/else logic
  def determineMNicheIndexBasedOnMorphologicalCriteria(individual: String):Int = {
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

    //We can find neighbourhood in parallel and we don't need evaluated population. We probably need to store all our individuals in the MAP for development convenience.
    val populationWithNeighbouthood = findNeighbours(evaluatedPopulation)

//    println(populationWithNeighbouthood.map(ind => ind.individual + "_" + ind.neighbours.map(_.score.toString).mkString(",")).mkString("\n"))

//    val accommodatedPopulation = evaluatedPopulation.map(evaluatedInd =>
//      EvaluatedStrWithAccommodation(evaluatedInd.individual, evaluatedInd.score, determineMNicheIndexBasedOnMorphologicalCriteria(evaluatedInd.individual))
//    )

    val evaluatedBasedOnNiches = evaluateLocalPerformances(populationWithNeighbouthood)

    println(evaluatedBasedOnNiches.map(ind => ind.individual + "_" + ind.localScore).mkString("\n"))
    println("\n\n ============================================ \n")


    val selectedParents = evaluatedBasedOnNiches.sortWith(_.localScore > _.localScore).take(evaluatedPopulation.length / 2)

    val offspring: Seq[String] = selectedParents.map(parent => mutateIndividual(parent.individual.individual))


    val populationWithOffspring = evaluatedBasedOnNiches ++ evaluateAndGetNichesPerformances(offspring)

    val newGenaration = populationWithOffspring.sortWith(_.localScore > _.localScore).take(population.length)
//    println(newGenaration.mkString(","))

    newGenaration.map(_.individual.individual)
  }

  //WE can move it to function that computes niche based scores

  def evaluateAndGetNichesPerformances(individuals: Seq[String]): Seq[EvaluatedStrWithLocalScores] = {
    val evaluatedByGlobalObjectiveOffspring = evaluatePopulation(individuals)
    val withNegboursOffspring = findNeighbours(evaluatedByGlobalObjectiveOffspring)
    val evaluatedBasedOnNichesOffspring = evaluateLocalPerformances(withNegboursOffspring)
    evaluatedBasedOnNichesOffspring
  }

  //Performance is suboptimal for now
  def findNeighbours(population: Seq[EvaluatedStr]):Seq[EvaluatedStrWithNeighbours] = {
    population.map{ind =>
      val distanceToNeighbours = mutable.Buffer[DistanceToNeighbour]()
      for(another <- population.diff(ind.individual)) {
        distanceToNeighbours.append(distance(ind, another))
      }
      val neighbours = distanceToNeighbours.sortWith(_.distance < _.distance).take(neighbourhoodSize).map(_.neighbour)
      EvaluatedStrWithNeighbours(ind, neighbours)
    }
  }

  def mutateIndividual(individual: String):String = {
    val randomInt = Random.nextInt(10)
    val randomIndex = Random.nextInt(individual.length)
    individual.map(_.toString).updated(randomIndex, randomInt.toString).mkString
  }

}

  case class DistanceToNeighbour(neighbour: EvaluatedStr, distance: Int)


case class EvaluatedStr(individual: String, score: Int)
case class EvaluatedStrWithNeighbours(individual: EvaluatedStr, neighbours: Seq[EvaluatedStr])
case class EvaluatedStrWithLocalScores(individual: EvaluatedStr, localScore: Int)

case class EvaluatedStrWithAccommodation(individual: String, score: Int, accommodation: Int)
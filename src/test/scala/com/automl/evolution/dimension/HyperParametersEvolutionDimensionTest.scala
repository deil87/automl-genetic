package com.automl.evolution.dimension

import akka.actor.ActorSystem
import com.automl.dataset.Datasets
import com.automl.evolution.dimension.hparameter._
import com.automl.population.HPPopulation
import com.automl.problemtype.ProblemType.MultiClassClassificationProblem
import com.automl.template.simple.Bayesian
import com.typesafe.scalalogging.LazyLogging
import org.scalatest.{FunSuite, Matchers}

import scala.collection.mutable
import scala.util.Random

class HyperParametersEvolutionDimensionTest extends FunSuite with Matchers with LazyLogging{
  implicit val system = ActorSystem("AutoMLSuite-system")
  implicit val logPaddingSize = 0

  val problem = MultiClassClassificationProblem
  val templateEvolutionDimension = new TemplateEvolutionDimension(None,1, problem)

  test("mutation of the group should work") {
    val hpGroupOfParameters =  BayesianHPGroup()
    val smoothing = hpGroupOfParameters.hpParameters
    val hpGroupOfParametersOrig = hpGroupOfParameters.mutate()

    val isAllParametersChanged = hpGroupOfParametersOrig.hpParameters.forall{ case param@Smoothing(_) =>
      param.currentValue != smoothing
    }
    isAllParametersChanged should be(true)
  }

  // TODO we can't guaranty that
  ignore("every evolution should improve performance") {
    val seed = new Random().nextLong()

    val dimension = new HyperParametersEvolutionDimension(templateEvolutionDimension, problemType = problem, seed = seed)
    val initialPopulation = new HPPopulation(Seq(
      HyperParametersField(Seq(BayesianHPGroup())),
      HyperParametersField(Seq(BayesianHPGroup()))
    ))

    val workingDF = Datasets.getGlassDataFrame(1234)

    logger.debug("Evolving population for the very first time...")
    dimension.evolve(initialPopulation, workingDF, null)

    val evolvedPopulation = dimension.getPopulation

    val bestIndividualAfterFirstEvolution = dimension.getLastEvaluatedPopulation(workingDF, null).sortWith(_.score > _.score).head

    logger.debug("Evolving population second time...")
    dimension.evolve(evolvedPopulation, workingDF, null)

    val bestIndividualAfterSecondEvolution = dimension.getLastEvaluatedPopulation(workingDF, null).sortWith(_.score > _.score).head

    logger.debug(s"Best individual before ${bestIndividualAfterFirstEvolution.field} with score ${bestIndividualAfterFirstEvolution.score}")
    logger.debug(s"Best individual after ${bestIndividualAfterSecondEvolution.field} with score ${bestIndividualAfterSecondEvolution.score}")
    bestIndividualAfterSecondEvolution.score > bestIndividualAfterFirstEvolution.score should be(true)
  }

  test("that evolution process will converge with smoothing == 1.0 ") {
    val seed = new Random().nextLong()

    val dimension = new HyperParametersEvolutionDimension(templateEvolutionDimension, problemType = problem, seed = seed)
    val initialPopulation = new HPPopulation(Seq(
      HyperParametersField(Seq(BayesianHPGroup())),
      HyperParametersField(Seq(BayesianHPGroup()))
    ))

    val workingDF = Datasets.getGlassDataFrame(1234)


    def repeatRecursively[T](population: T, fun: T => T , numberOfRepeats: Int):T = {
      if(numberOfRepeats > 0) {
        val evolvedPopulation: T = fun(population)
        repeatRecursively(evolvedPopulation, fun, numberOfRepeats - 1)
      } else population
    }

    val evolveFun: HPPopulation => HPPopulation = dimension.evolve(_, workingDF, null)
    repeatRecursively(initialPopulation, evolveFun, 15)

    val bestSmoothingValue = dimension.getBestFromHallOfFame.modelsHParameterGroups.head.hpParameters.filter { case Smoothing(_) => true }.head.currentValue

    bestSmoothingValue should be(1.0)
  }

  test("Check that hash code is correct for storing complex class in the hash map") {
    case class Property1(currentValue: Double)

    val map: mutable.Map[(ExpGroup, Long), Double] = mutable.Map[(ExpGroup, Long), Double]()

    case class ExpGroup(items: Seq[Property1]) {
      def calc = items.map(_.currentValue)
    }
    val datasize = 100L
    val item1 = ExpGroup(Seq(Property1(42), Property1(43)))

    val item1Value = 43.5
    map.put((item1, datasize),item1Value)

    val item2 = ExpGroup(Seq(Property1(42), Property1(45)))
    val item2Value = 45.5

    map.put((item2, datasize),item2Value)

    val cacheKey = (ExpGroup(Seq(Property1(42), Property1(43))), datasize)

    map(cacheKey) should be(item1Value)

    val cacheKey2 = (item2, datasize)

    map(cacheKey2) should be(item2Value)

    val item3 = ExpGroup(Seq(Property1(47), Property1(45)))
    val cacheKey3 = (item3, datasize)
    map.get(cacheKey3) should be (None)

    //Lets mutate with copy item1 and see if hashCode was changed. Expect not to retrieve any value from map.
    val item4 = item1.copy(items = item1.items.map(_.copy(currentValue = 66)))
    val cacheKey4 = (item4, datasize)
    map.get(cacheKey4) should be (None)


    val item5 = ExpGroup(Seq(Property1(66), Property1(66)))
    val item6Value = 66.5
    map.put((item5, datasize), item6Value)

    val item6 = item1.copy(items = item1.items.map(_.copy(currentValue = 66)))
    val cacheKey6 = (item6, datasize)
    map(cacheKey6) should be (item6Value)
  }

  test("Check that hash code is correct for storing complex class in the hash map (deeper case)") {
    case class Property2[T](currentValue: Seq[PropertyInner[_]])
    case class PropertyInner[T](deepValue: Double)

    val map: mutable.Map[(ExpGroup, Long), Double] = mutable.Map[(ExpGroup, Long), Double]()

    case class ExpGroup(items: Seq[Property2[PropertyInner[_]]]) {
      def calc = items.map(_.currentValue)
    }
    val datasize = 100L
    val item1 = ExpGroup(Seq(Property2(Seq(PropertyInner(42),PropertyInner(43))), Property2(Seq(PropertyInner(44), PropertyInner(45)))))
    val item1Value = 66.5


    map.put((item1, datasize),item1Value)

    val item2 = ExpGroup(Seq(Property2(Seq(PropertyInner(42),PropertyInner(43))), Property2(Seq(PropertyInner(42), PropertyInner(43)))))
    val item2Value = 88.5
    map.put((item2, datasize), item2Value)

//    val item3 = item1.copy(items = item1.items.map(_.copy(currentValue = Seq(PropertyInner(42), PropertyInner(43)))))
    val item3 = item1.copy(items = Seq(Property2(Seq(PropertyInner(42),PropertyInner(43))), Property2(Seq(PropertyInner(42), PropertyInner(43)))))
    val cacheKey3 = (item3, datasize)
    map(cacheKey3) should be (item2Value)

    val cacheKey1 = (item1, datasize)
    map(cacheKey1) should be (item1Value)
  }
}

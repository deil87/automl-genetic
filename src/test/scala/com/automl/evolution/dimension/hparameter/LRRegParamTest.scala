package com.automl.evolution.dimension.hparameter

import akka.actor.ActorSystem
import com.automl.dataset.Datasets
import com.automl.evolution.dimension.TemplateEvolutionDimension
import com.automl.population.HPPopulation
import com.automl.problemtype.ProblemType.MultiClassClassificationProblem
import com.typesafe.scalalogging.LazyLogging
import org.scalatest.{FunSuite, Matchers}

import scala.util.Random

class LRRegParamTest extends FunSuite with Matchers with LazyLogging{

  implicit val system = ActorSystem("AutoMLSuite-system")
  implicit val logPaddingSize = 0


  val problem = MultiClassClassificationProblem
  val templateEvolutionDimension = new TemplateEvolutionDimension(None,1, problem)

  //TODO FIX TEST
  ignore("that evolution process will converge with regularisation param ") {
    val seed = new Random().nextLong()

    val dimension = new TemplateHyperParametersEvolutionDimension(templateEvolutionDimension, problemType = problem, seed = seed)
    val initialPopulation = new HPPopulation(Seq(
      HyperParametersField(Seq(LogisticRegressionHPGroup())),
      HyperParametersField(Seq(LogisticRegressionHPGroup())),
      HyperParametersField(Seq(LogisticRegressionHPGroup()))
    ))

    val workingDF = Datasets.getGlassDataFrame(1234)


    def repeatRecursively[T](population: T, fun: T => T , numberOfRepeats: Int):T = {
      if(numberOfRepeats > 0) {
        val evolvedPopulation: T = fun(population)
        repeatRecursively(evolvedPopulation, fun, numberOfRepeats - 1)
      } else population
    }

    val evolveFun: HPPopulation => HPPopulation = dimension.evolve(_, workingDF)
    repeatRecursively(initialPopulation, evolveFun, 15)

    val bestRegParamValue = dimension.getBestFromHallOfFame.modelsHParameterGroups.head.hpParameters.filter { case LRRegParam(_) => true }.head.currentValue

    bestRegParamValue should be(0.3)
  }

  //TODO does not test anything
  test("rounding for default values") {

    1 to 10 foreach(i => logger.info("LRRegParam rounded to: " + LRRegParam().round(new Random().nextDouble(), 1)))
  }

}

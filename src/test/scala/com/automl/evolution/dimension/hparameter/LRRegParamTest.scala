package com.automl.evolution.dimension.hparameter

import com.automl.dataset.Datasets
import com.automl.problemtype.ProblemType.MultiClassClassificationProblem
import com.typesafe.scalalogging.LazyLogging
import org.scalatest.{FunSuite, Matchers}

import scala.util.Random

class LRRegParamTest extends FunSuite with Matchers with LazyLogging{

  test("that evolution process will converge with regularisation param ") {

    val dimension = new TemplateHyperParametersEvolutionDimension(problemType = MultiClassClassificationProblem)
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

    val bestRegParamValue = dimension.getBestFromHallOfFame.field.modelsHParameterGroups.head.hpParameters.filter { case LRRegParam() => true }.head.currentValue

    bestRegParamValue should be(0.3)
  }

  test("rounding for default values") {

    1 to 1000 foreach(i => println(LRRegParam().round(new Random().nextDouble(), 1)))
  }

}

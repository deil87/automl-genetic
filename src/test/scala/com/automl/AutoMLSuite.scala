package com.automl

import akka.actor.ActorSystem
import com.automl.evolution.dimension.{TemplateEvolutionDimension, TemplateHyperParametersEvolutionDimension}
import com.automl.evolution.diversity.DistinctDiversityStrategy
import com.automl.evolution.mutation.DepthDependentTemplateMutationStrategy
import com.automl.helper.{FitnessResult, PopulationHelper}
import com.automl.spark.SparkSessionProvider
import com.automl.template._
import com.automl.template.simple._
import org.apache.spark.ml.feature.VectorAssembler
import org.scalatest.{Matchers, WordSpec}
import utils.SparkMLUtils


class AutoMLSuite extends WordSpec with Matchers with SparkSessionProvider {

  import utils.SparkMLUtils._

  implicit val system = ActorSystem("AutoMLSuite-system")

  "AutoML" should {

    "mutate templateTree from base model to complex algorithm" ignore {

      val seed: Seq[LeafTemplate[SimpleModelMember]] = Seq(
        LeafTemplate(Bayesian()),
        LeafTemplate(LinearRegressionModel()),
        LeafTemplate(DecisionTree())
      )

      val seedPopulation = new Population(seed)

      val population = Population.fromSeedPopulation(seedPopulation).withSize(10).build

      val autoMl = new AutoML(null, 50000, useMetaDB = false, initialPopulationSize = Some(10))

      PopulationHelper.print(population)

      val distinctStrategy = new DistinctDiversityStrategy()
      
      val mutationStrategy = new DepthDependentTemplateMutationStrategy(distinctStrategy)

      val mutated = mutationStrategy.mutate(population)

      PopulationHelper.print(mutated)

      val mutated2 = mutationStrategy.mutate(mutated)
      PopulationHelper.print(mutated2)

      val mutated3 = mutationStrategy.mutate(mutated2)
      PopulationHelper.print(mutated3)

      //TODO make mutation happens every time
      mutated shouldNot be(population)
      mutated2 shouldNot be(mutated)
      mutated3 shouldNot be(mutated2)
    }

    "find best template with most optimal fitness value" in {

      val seed: Seq[LeafTemplate[SimpleModelMember]] = Seq(
        LeafTemplate(LinearRegressionModel()),
        LeafTemplate(Bayesian()),
        LeafTemplate(GradientBoosting()),
        LeafTemplate(DecisionTree())
      )

      val seedPopulation = new Population(seed)

      val population = Population.fromSeedPopulation(seedPopulation).withSize(10).build

      val airlineDF = SparkMLUtils.loadParquet("src/test/resources/airline_allcolumns_sampled_100k_parquet")
        .select("DayOfWeek", "Distance", "DepTime", "CRSDepTime", "DepDelay")

      val features = Array("Distance", "DayOfWeek")
      val oheFeatures = Array.empty

      val combinedFeatures = features

      val featuresColName: String = "features"

      def featuresAssembler = {
        new VectorAssembler()
          .setInputCols(combinedFeatures)
          .setOutputCol(featuresColName)
      }
      import org.apache.spark.sql.functions.monotonically_increasing_id

      val prepairedAirlineDF = airlineDF
        .limit(15000)
        .applyTransformation(featuresAssembler)
        .withColumnRenamed("DepDelay", "label")
        .toDouble("label")
        .filterOutNull("label")
        .withColumn("uniqueIdColumn", monotonically_increasing_id)
        .showN_AndContinue(10)
        .cache()

      val Array(trainingSplit, testSplit) = prepairedAirlineDF.randomSplit(Array(0.8, 0.2))

      trainingSplit.cache()

      val autoMl = new AutoML(trainingSplit, 300000, useMetaDB = false, initialPopulationSize = Some(7), seedPopulation = seedPopulation, maxGenerations = 5)

      val templateEvDim = new TemplateEvolutionDimension
      val hyperParamsEvDim = new TemplateHyperParametersEvolutionDimension

      autoMl.runEvolution(templateEvDim, hyperParamsEvDim)

    }

  }
}


package com.automl.evolution.dimension

import akka.actor.ActorSystem
import com.automl.{AutoML, TPopulation, PopulationEvaluator}
import com.automl.spark.SparkSessionProvider
import com.automl.template.ensemble.bagging.Bagging
import com.automl.template.{LeafTemplate, NodeTemplate}
import com.automl.template.simple.{Bayesian, DecisionTree, LinearRegressionModel, SimpleModelMember}
import org.apache.spark.ml.feature.VectorAssembler
import org.scalatest.{Matchers, WordSpec}
import utils.{LabeledVector, SparkMLUtils}

class TemplateEvolutionDimensionSuite extends WordSpec with Matchers with SparkSessionProvider {
  import ss.implicits._
  import utils.SparkMLUtils._

  trait Fixture {

    implicit val system = ActorSystem("AutoMLSuite-system")

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

    val preparedAirlineDF = airlineDF
      .limit(5000)
      .applyTransformation(featuresAssembler)
      .withColumnRenamed("DepDelay", "label")
      .toDouble("label")
      .filterOutNull("label")
      .withColumn("uniqueIdColumn", monotonically_increasing_id)
//      .showN_AndContinue(10)
      .cache()

    val Array(trainingSplit, testSplit) = preparedAirlineDF.randomSplit(Array(0.8, 0.2))

    trainingSplit.cache()

  }

  "TemplateEvolutionDimension" should {

    "use different cache values for same template but different dataframe" in new Fixture {
      val seed: Seq[LeafTemplate[SimpleModelMember]] = Seq(
        LeafTemplate(Bayesian()),
        LeafTemplate(LinearRegressionModel()),
        LeafTemplate(DecisionTree())
      )

      val population = new TPopulation(seed)

      val ds2 = trainingSplit.limit(20)
      val ds3 = trainingSplit.limit(300)

      val t = new TemplateEvolutionDimension()

      t.evaluate(population, ds2, Map.empty)
      t.evaluate(population, ds3, Map.empty)

      t.templatesEvaluationCache.size should be (6)  // (numbers of templates in population) * (# of different sizes of training datasets)
    }

    "caching is working within ensemble nodes" in new Fixture{

      val template = Seq(
        NodeTemplate(Bagging(), Seq(
          LeafTemplate(LinearRegressionModel()),
          NodeTemplate(Bagging(), Seq(
            LeafTemplate(LinearRegressionModel()),
            LeafTemplate(LinearRegressionModel())
          ))
        )
        )
      )

      val t = new TemplateEvolutionDimension()

      val testPopulation = new TPopulation(template)

      val ds2 = trainingSplit.limit(20)
      t.evaluate(testPopulation, ds2, Map.empty).nonEmpty shouldBe true
    }

  }
}


package com.automl.evolution.dimension

import akka.actor.ActorSystem
import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.MultiClassClassificationProblem
import com.automl.{AutoML, ConfigProvider}
import com.automl.dataset.Datasets
import com.automl.evolution.dimension.hparameter.{BayesianHPGroup, MaxDepthRF, RandomForestHPGroup, Smoothing}
import com.automl.population.TPopulation
import com.automl.spark.SparkSessionProvider
import com.automl.template.{LeafTemplate, NodeTemplate}
import com.automl.template.simple._
import org.apache.spark.ml.feature.VectorAssembler
import org.scalatest.{Matchers, WordSpec}
import utils.SparkMLUtils

class TemplateEvolutionDimensionSuite extends WordSpec with Matchers with SparkSessionProvider {
  import ss.implicits._
  import utils.SparkMLUtils._

  trait Fixture {

    implicit val system = ActorSystem("AutoMLSuite-system")
    implicit val logPaddingSize = 0
  }

  "TemplateEvolutionDimension" should {

    "use different cache values for same template but different dataframes" in new Fixture {
      ConfigProvider.clearOverride.addOverride(
        """
          |evolution {
          |  hyperParameterDimension {
          |    enabled = false
          |  }
          |}
        """)
      val seedPopulation: Seq[LeafTemplate[SimpleModelMember]] = Seq(
        LeafTemplate(Bayesian()),
        LeafTemplate(DecisionTree())
      )

      val population = new TPopulation(seedPopulation)

      val iridData = Datasets.getIrisDataFrame(1234)
      val ds15 = iridData.limit(15)
      val ds20 = iridData.limit(20)

      val problemType = MultiClassClassificationProblem
      val t = new TemplateEvolutionDimension(initialPopulation = Some(population), problemType = problemType)

      t.evaluatePopulation(population, ds15, null)
      t.evaluatePopulation(population, ds20, null)

      t.individualsEvaluationCacheExtended.size should be (4)  // (numbers of templates in population) * (# of different sizes of training datasets)
    }

    "use different cache values for same templates but different hpGroups but same dataframes" in new Fixture {
      ConfigProvider.clearOverride.addOverride(
        """
          |evolution {
          |  hyperParameterDimension {
          |    enabled = false
          |  }
          |}
        """)

      val seedPopulation: Seq[LeafTemplate[SimpleModelMember]] = Seq(
        LeafTemplate(Bayesian(BayesianHPGroup(Seq(Smoothing(Some(3.0)))))),
        LeafTemplate(RandomForest(RandomForestHPGroup(Seq(MaxDepthRF(Some(6.0))))))
      )

      val seedPopulation2: Seq[LeafTemplate[SimpleModelMember]] = Seq(
        LeafTemplate(Bayesian(BayesianHPGroup(Seq(Smoothing(Some(5.0)))))),
        LeafTemplate(RandomForest(RandomForestHPGroup(Seq(MaxDepthRF(Some(3.0))))))
      )

      val population = new TPopulation(seedPopulation)

      val iridData = Datasets.getIrisDataFrame(1234)
      val ds300 = iridData.limit(300)

      val problemType = MultiClassClassificationProblem
      val t = new TemplateEvolutionDimension(initialPopulation = None, problemType = problemType)

      t.evaluatePopulation(new TPopulation(seedPopulation), ds300, null)
      t.evaluatePopulation(new TPopulation(seedPopulation2), ds300, null)

      // Following line should not bring new records to the cache
      t.evaluatePopulation(new TPopulation(seedPopulation2), ds300, null)

      t.individualsEvaluationCacheExtended.size should be (4)  // (numbers of templates in population) * (# of different sizes of training datasets)
    }

    "caching is working within ensemble nodes" ignore new Fixture{
      ConfigProvider.clearOverride.addOverride(
        """
          |evolution {
          |  hyperParameterDimension {
          |    enabled = false
          |  }
          |}
        """)

      val template =
        NodeTemplate(SparkGenericBagging(), Seq(
          LeafTemplate(LogisticRegressionModel()),
          NodeTemplate(SparkGenericBagging(), Seq(
            LeafTemplate(LogisticRegressionModel()),
            LeafTemplate(DecisionTree())
          ))
        ))
      val problemType = ProblemType.MultiClassClassificationProblem

      val t = new TemplateEvolutionDimension(problemType = problemType)

      val testPopulation = new TPopulation(Seq(template))

      val iridData = Datasets.getIrisDataFrame(1234)
      val ds2 = iridData.limit(20)

      t.evaluatePopulation(testPopulation, ds2, null).nonEmpty shouldBe true

      private val key =  ??? //(template, template.internalHyperParamsMap, ds2.count())
      t.individualsEvaluationCacheExtended.get(key).nonEmpty shouldBe true
    }

  }
}


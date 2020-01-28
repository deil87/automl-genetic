package com.automl.evolution.dimension

import akka.actor.ActorSystem
import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.MultiClassClassificationProblem
import com.automl.{AutoML, ConfigProvider, EvaluatedTemplateData}
import com.automl.dataset.Datasets
import com.automl.evolution.dimension.hparameter._
import com.automl.helper.FitnessResult
import com.automl.population.TPopulation
import com.automl.spark.SparkSessionProvider
import com.automl.template.{LeafTemplate, NodeTemplate}
import com.automl.template.simple._
import org.apache.spark.ml.feature.VectorAssembler
import org.scalatest.{Matchers, WordSpec}
import com.test.TestBase

class TemplateEvolutionDimensionSuite extends WordSpec with SparkSessionProvider with TestBase{
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

    "use same cache values for essentially same templates (different order of sub-members)" in new Fixture {
      ConfigProvider.clearOverride.addOverride(
        """
          |evolution {
          |  hyperParameterDimension {
          |    enabled = false
          |  }
          |}
        """)

      val seed = 1234
      val seedPopulation: Seq[LeafTemplate[SimpleModelMember]] = Seq(
        LeafTemplate(Bayesian(BayesianHPGroup(Seq(Smoothing(Some(3.0)))))),
        LeafTemplate(RandomForest(RandomForestHPGroup(Seq(MaxDepthRF(Some(6.0)))), seed))
      )

      val seedPopulation2: Seq[LeafTemplate[SimpleModelMember]] = Seq(
        LeafTemplate(RandomForest(RandomForestHPGroup(Seq(MaxDepthRF(Some(6.0)))), seed)),
        LeafTemplate(Bayesian(BayesianHPGroup(Seq(Smoothing(Some(3.0))))))
      )

      val iridData = Datasets.getIrisDataFrame(1234)
      val ds20 = iridData.limit(20)

      val problemType = MultiClassClassificationProblem
      val t = new TemplateEvolutionDimension(initialPopulation = None, problemType = problemType)

      t.evaluatePopulation(new TPopulation(seedPopulation), ds20, null)
      t.evaluatePopulation(new TPopulation(seedPopulation2), ds20, null)


      t.individualsEvaluationCacheExtended.size should be (2)
    }

    "use different cache records if we have different samples( count does not guaranty much)" in new Fixture {
      ConfigProvider.clearOverride.addOverride(
        """
          |evolution {
          |  hyperParameterDimension {
          |    enabled = false
          |  }
          |}
        """)

      val seed = 1234
      val seedPopulation: Seq[LeafTemplate[SimpleModelMember]] = Seq(
        LeafTemplate(Bayesian(BayesianHPGroup(Seq(Smoothing(Some(3.0))))))
      )

      val iridData = Datasets.getIrisDataFrame(1234)
      val ds20 = iridData.limit(20)
      val iridData2 = Datasets.getIrisDataFrame(5678)
      val ds20_another_sample = iridData2.limit(20)

      val problemType = MultiClassClassificationProblem
      val t = new TemplateEvolutionDimension(initialPopulation = None, problemType = problemType)

      val code1 = ds20.hashCode()
      val code2 = ds20.hashCode()

      code1 shouldBe code2

      val code3 = ds20_another_sample.hashCode()
      code1 shouldNot be(code3)


      t.evaluatePopulation(new TPopulation(seedPopulation), ds20, null)
      t.evaluatePopulation(new TPopulation(seedPopulation), ds20, null)
      t.evaluatePopulation(new TPopulation(seedPopulation), ds20_another_sample, null)

      t.individualsEvaluationCacheExtended.size should be (2)
    }

    "hash codes for all algos are consistent ( need to make sure that seeds are set)" in {
      val bayesian = LeafTemplate(Bayesian(BayesianHPGroup(Seq(Smoothing(Some(3.0))))))
      val bayesian2 = LeafTemplate(Bayesian(BayesianHPGroup(Seq(Smoothing(Some(3.0))))))

      bayesian.hashCode() shouldBe bayesian2.hashCode()

      val dt = LeafTemplate(DecisionTree(DecisionTreeHPGroup(Seq(MaxDepth(Some(6.0)))), 1234))
      val dt2 = LeafTemplate(DecisionTree(DecisionTreeHPGroup(Seq(MaxDepth(Some(6.0)))), 1234))

      dt.hashCode() shouldBe dt2.hashCode()

      val rf = LeafTemplate(RandomForest(RandomForestHPGroup(Seq(MaxDepthRF(Some(6.0)))), 12345))
      val rf2 = LeafTemplate(RandomForest(RandomForestHPGroup(Seq(MaxDepthRF(Some(6.0)))), 12345))

      rf.hashCode() shouldBe rf2.hashCode()
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

    "hall of fame is being updated properly depending on the metric( f1 case)" in new Fixture {

      val population = new TPopulation(Seq(
        LeafTemplate(Bayesian()),
        LeafTemplate(DecisionTree())
      ))

      ConfigProvider.clearOverride.addOverride(
        """
          |evolution {
          |  evaluation {
          |    multiclass.metric = "f1"
          |  }
          |}
        """)

      private val problemType: ProblemType.MultiClassClassificationProblem.type = MultiClassClassificationProblem
      val evaluatedAlreadyInHOF = EvaluatedTemplateData("id0", null, null, FitnessResult(Map("f1" -> 0.5, "logloss" -> 0.45), problemType, null))
      val evaluatedA = EvaluatedTemplateData("id1", LeafTemplate(Bayesian()), null, FitnessResult(Map("f1" -> 0.2, "logloss" -> 0.3), problemType, null))
      val evaluatedB = EvaluatedTemplateData("id2", LeafTemplate(Bayesian()), null, FitnessResult(Map("f1" -> 0.3, "logloss" -> 0.2), problemType, null))
      val evaluatedC = EvaluatedTemplateData("id3", LeafTemplate(Bayesian()), null, FitnessResult(Map("f1" -> 0.1, "logloss" -> 1.2), problemType, null))
      val evaluatedD = EvaluatedTemplateData("id4", LeafTemplate(Bayesian()), null, FitnessResult(Map("f1" -> 0.6, "logloss" -> 0.5), problemType, null))
      val evaluatedE = EvaluatedTemplateData("id5", LeafTemplate(Bayesian()), null, FitnessResult(Map("f1" -> 0.8, "logloss" -> 0.7), problemType, null))
      val evaluatedF = EvaluatedTemplateData("id6", LeafTemplate(Bayesian()), null, FitnessResult(Map("f1" -> 0.9, "logloss" -> 0.1), problemType, null))


      val ted = new TemplateEvolutionDimension(initialPopulation = Some(population), problemType = problemType)

      ted.hallOfFame += evaluatedAlreadyInHOF // otherwise if queue is empty we would just take N elements

      ted.hallOfFame.size shouldBe 1

      ted.updateHallOfFame(Seq(evaluatedA, evaluatedB, evaluatedC, evaluatedD, evaluatedE, evaluatedF))

      ted.hallOfFame.size shouldBe 4 // as only 3 items are better than `evaluatedAlreadyInHOF`
      ted.hallOfFame should not contain allElementsOf( Seq(evaluatedA, evaluatedB, evaluatedC))

    }

    "hall of fame is being updated properly depending on the metric( logloss case)" in new Fixture {

      val population = new TPopulation(Seq(
        LeafTemplate(Bayesian()),
        LeafTemplate(DecisionTree())
      ))

      ConfigProvider.clearOverride.addOverride(
        """
          |evolution {
          |  evaluation {
          |    multiclass.metric = "logloss"
          |  }
          |}
        """)

      private val problemType: ProblemType.MultiClassClassificationProblem.type = MultiClassClassificationProblem
      val evaluatedAlreadyInHOF = EvaluatedTemplateData("id0", null, null, FitnessResult(Map("f1" -> 0.5, "logloss" -> 0.45), problemType, null))
      val evaluatedA = EvaluatedTemplateData("id1", LeafTemplate(Bayesian()), null, FitnessResult(Map("f1" -> 0.2, "logloss" -> 0.3), problemType, null))
      val evaluatedB = EvaluatedTemplateData("id2", LeafTemplate(Bayesian()), null, FitnessResult(Map("f1" -> 0.3, "logloss" -> 0.2), problemType, null))
      val evaluatedC = EvaluatedTemplateData("id3", LeafTemplate(Bayesian()), null, FitnessResult(Map("f1" -> 0.1, "logloss" -> 1.2), problemType, null))
      val evaluatedD = EvaluatedTemplateData("id4", LeafTemplate(Bayesian()), null, FitnessResult(Map("f1" -> 0.6, "logloss" -> 0.5), problemType, null))
      val evaluatedE = EvaluatedTemplateData("id5", LeafTemplate(Bayesian()), null, FitnessResult(Map("f1" -> 0.8, "logloss" -> 0.46), problemType, null))
      val evaluatedF = EvaluatedTemplateData("id6", LeafTemplate(Bayesian()), null, FitnessResult(Map("f1" -> 0.9, "logloss" -> 0.66), problemType, null))


      val ted = new TemplateEvolutionDimension(initialPopulation = Some(population), problemType = problemType)

      ted.hallOfFame += evaluatedAlreadyInHOF // otherwise if queue is empty we would just take N elements

      ted.hallOfFame.size shouldBe 1

      ted.updateHallOfFame(Seq(evaluatedA, evaluatedB, evaluatedC, evaluatedD, evaluatedE, evaluatedF))

      ted.hallOfFame.size shouldBe 3 // as only 2 items are better than `evaluatedAlreadyInHOF`
      ted.hallOfFame should not contain allElementsOf( Seq(evaluatedC, evaluatedD, evaluatedE, evaluatedF))

    }

  }
}


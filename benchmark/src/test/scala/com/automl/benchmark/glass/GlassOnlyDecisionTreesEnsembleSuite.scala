package com.automl.benchmark.glass

import akka.actor.ActorSystem
import com.automl.spark.SparkSessionProvider
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpec}


class GlassOnlyDecisionTreesEnsembleSuite extends WordSpec with Matchers with BeforeAndAfterAll with SparkSessionProvider {

  implicit val system = ActorSystem("AutoML-system")


  "AutoML on glass dataset" should {


    "consistently find decision trees D1 with specific set of hyper-parameters as a winner" in {

      val benchmark = new GlassOnlyDecisionTreesEnsembleRuns().depthOneRun()


    }

    "consistently find similar ensembles D2 with specific set of hyper-parameters as a winner" in {

      val benchmark = new GlassOnlyDecisionTreesEnsembleRuns().depthTwoRun()


    }

    "consistently find similar ensembles D3 with specific set of hyper-parameters as a winner" in {

      val benchmark = new GlassOnlyDecisionTreesEnsembleRuns().depthThreeRun()


    }
  }

}





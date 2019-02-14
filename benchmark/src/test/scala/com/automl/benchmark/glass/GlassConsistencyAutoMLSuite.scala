package com.automl.benchmark.glass

import akka.actor.ActorSystem
import com.automl.spark.SparkSessionProvider
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpec}


class GlassConsistencyAutoMLSuite extends WordSpec with Matchers with BeforeAndAfterAll with SparkSessionProvider {

  implicit val system = ActorSystem("AutoML-system")


  "AutoML on glass dataset" should {


    "consistently find decision trees with specific set of hyper-parameters as a winner" in {

      val benchmark = new GlassDataSetConsistencyBenchmark().depthOneRun()


    }
  }

}





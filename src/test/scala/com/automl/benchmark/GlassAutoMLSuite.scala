package com.automl.benchmark

import akka.actor.ActorSystem
import com.automl.spark.SparkSessionProvider
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpec}
import utils.TestBase


class GlassAutoMLSuite extends WordSpec with BeforeAndAfterAll with SparkSessionProvider with TestBase{

  implicit val system = ActorSystem("AutoML-system")    //TODO This `as` stuff should probably be inside AutoML object?


  "AutoML benchmark on glass dataset" should {

    "find best template with most optimal fitness value" ignore {

      val benchmark = new GlassDataSetBenchmark().run()
    }
  }

}





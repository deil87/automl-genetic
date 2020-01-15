package com.automl.benchmark

import akka.actor.ActorSystem
import com.automl.spark.SparkSessionProvider
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpec}


class CarAutoMLSuite extends WordSpec with Matchers with BeforeAndAfterAll with SparkSessionProvider {

  implicit val system = ActorSystem("AutoML-system")    //TODO This as stuff should probably be inside AutoML object?


  "AutoML on car dataset" should {


    "find best template with most optimal fitness value" in {

      val benchmark = new CarDataSetBenchmark().run()
    }
  }

}
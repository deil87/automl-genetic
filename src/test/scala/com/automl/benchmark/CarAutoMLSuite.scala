package com.automl.benchmark

import akka.actor.ActorSystem
import com.automl.spark.SparkSessionProvider
import org.scalatest.{BeforeAndAfterAll, Ignore, Matchers, WordSpec}
import com.test.TestBase

@Ignore
class CarAutoMLSuite extends WordSpec with BeforeAndAfterAll with SparkSessionProvider with TestBase{

  implicit val system = ActorSystem("AutoML-system")    //TODO This `as` stuff should probably be inside AutoML object?


  "AutoML benchmark on car dataset" should {

    "find best template with most optimal fitness value" in {

      val benchmark = new CarDataSetBenchmark().run()
    }
  }

}
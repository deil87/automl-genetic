package com.automl.benchmark.balance_scale

import akka.actor.ActorSystem
import com.automl.spark.SparkSessionProvider
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpec}


class BalanceAutoMLSuite extends WordSpec with Matchers with BeforeAndAfterAll with SparkSessionProvider {

  implicit val system = ActorSystem("AutoML-system")    //TODO This as stuff should probably be inside AutoML object?


  "AutoML on balance dataset" should {


    "find best template with most optimal fitness value" in {

      new BalanceScaleDataSetBenchmark().run()


    }
  }

}





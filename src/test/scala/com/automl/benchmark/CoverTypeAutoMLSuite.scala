package com.automl.benchmark

import akka.actor.ActorSystem
import com.automl.spark.SparkSessionProvider
import com.test.TestBase
import org.scalatest.time.{Minutes, Span}
import org.scalatest.{BeforeAndAfterAll, Ignore, WordSpec}

@Ignore
class CoverTypeAutoMLSuite extends WordSpec with BeforeAndAfterAll with SparkSessionProvider with TestBase{

  implicit val system = ActorSystem("AutoML-system")    //TODO This `as` stuff should probably be inside AutoML object?

//  override val timeLimit: Span = Span(31, Minutes)

  "AutoML benchmark on covtype dataset" should {

    "find best template with most optimal fitness value" in {

      val benchmark = new CoverTypeDataSetBenchmark().run()
    }
  }

}
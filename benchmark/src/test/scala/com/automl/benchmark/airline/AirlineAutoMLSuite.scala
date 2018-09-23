package com.automl.benchmark.airline

import akka.actor.{ActorSystem, Props}
import akka.http.scaladsl.Http
import akka.stream.ActorMaterializer
import com.automl.route.{WebClientNotifierActor, StaticResourcesRoute, WebSocketServiceRoute}
import com.automl.spark.SparkSessionProvider
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpec}

import akka.http.scaladsl.server.Directives._


class AirlineAutoMLSuite extends WordSpec with Matchers with BeforeAndAfterAll with SparkSessionProvider {

  implicit val system = ActorSystem("AutoML-system")    //TODO This as stuff should probably be inside AutoML object?
  implicit val materializer = ActorMaterializer()

  implicit val executionContext = system.dispatcher

  val webClientNotifier = system.actorOf(Props(classOf[WebClientNotifierActor]), "webClientNotifier")

  val staticRoute = new StaticResourcesRoute().staticResources
  val wsRoute = new WebSocketServiceRoute(webClientNotifier).route


  val port = 8088
  val bindingFuture = Http().bindAndHandle(staticRoute ~ wsRoute, "localhost", port)

  println(s"Server online at http://localhost:$port/automl/\nPress RETURN to stop...")


  "AutoML on Airline dataset" should {


    "find best template with most optimal fitness value" in {

      val benchmark = new AirlineDataSetBenchmark().run()


    }

  }

  override protected def afterAll(): Unit = {
    bindingFuture
      .flatMap(_.unbind()) // trigger unbinding from the port
      .onComplete(_ => system.terminate()) // and shutdown when done
  }
}


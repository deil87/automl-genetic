package com.automl
import java.util.concurrent.TimeUnit

import akka.actor.{ActorSystem, Props}
import akka.http.scaladsl.Http
import akka.http.scaladsl.server.Directives._
import akka.stream.ActorMaterializer
import akka.util.Timeout
import com.automl.route.{ClientConnectionActor, WebClientNotifierActor, StaticResourcesRoute, WebSocketServiceRoute}

import scala.io.StdIn
import scala.util.{Failure, Success}

object AutoMLWeb extends App {


  implicit val system = ActorSystem("AutoML-system")
  implicit val materializer = ActorMaterializer()

  implicit val executionContext = system.dispatcher

  val notifier = system.actorOf(Props(classOf[WebClientNotifierActor]))

  val staticRoute = new StaticResourcesRoute().staticResources
  val wsRoute = new WebSocketServiceRoute(notifier).route


  private val port = 8088
  val bindingFuture = Http().bindAndHandle(staticRoute ~ wsRoute, "localhost", port)

  println(s"Server online at http://localhost:$port/automl/\nPress RETURN to stop...")
  StdIn.readLine() // let it run until user presses return
  bindingFuture
    .flatMap(_.unbind()) // trigger unbinding from the port
    .onComplete(_ => system.terminate()) // and shutdown when done
}

package com.automl
import akka.actor.ActorSystem
import akka.http.scaladsl.Http
import akka.http.scaladsl.server.Directives._
import akka.stream.ActorMaterializer
import com.automl.route.{StaticResourcesRoute, WebSocketServiceRoute}

import scala.io.StdIn

object AutoMLWeb extends App {


  implicit val system = ActorSystem("AutoML-system")
  implicit val materializer = ActorMaterializer()

  implicit val executionContext = system.dispatcher

  val staticRoute = new StaticResourcesRoute().staticResources
  val wsRoute = new WebSocketServiceRoute().route

  private val port = 8088
  val bindingFuture = Http().bindAndHandle(staticRoute ~ wsRoute, "localhost", port)

  println(s"Server online at http://localhost:$port/automl/\nPress RETURN to stop...")
  StdIn.readLine() // let it run until user presses return
  bindingFuture
    .flatMap(_.unbind()) // trigger unbinding from the port
    .onComplete(_ => system.terminate()) // and shutdown when done
}

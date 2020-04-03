package com.automl
import java.net.URI

import akka.actor.{ActorSystem, Props}
import akka.http.scaladsl.Http
import akka.http.scaladsl.server.Directives._
import akka.stream.ActorMaterializer
import com.automl.benchmark.{CarDataSetBenchmark, GlassDataSetBenchmark}
import com.automl.route.{ClientConnectionActor, StaticResourcesRoute, WebClientNotifierActor, WebSocketServiceRoute}


object AutoMLWebUI extends App {


  implicit val system = ActorSystem("AutoML-system")
  implicit val materializer = ActorMaterializer()

  implicit val executionContext = system.dispatcher

  val notifier = system.actorOf(Props(classOf[WebClientNotifierActor]), "webClientNotifier")

  val staticRoute = new StaticResourcesRoute().staticResources
  val wsRoute = new WebSocketServiceRoute(notifier).route


  private val port = 8098
  val bindingFuture = Http().bindAndHandle(staticRoute ~ wsRoute, "localhost", port)
  val webInterfaceURL = s"http://localhost:$port/automl/"
  println(s"Server online at $webInterfaceURL\n Press RETURN to stop...")

  import java.awt.Desktop

  if (Desktop.isDesktopSupported && Desktop.getDesktop.isSupported(Desktop.Action.BROWSE))
    Desktop.getDesktop.browse(new URI(webInterfaceURL))

//  val benchmark = new GlassDataSetBenchmark().run()
  val benchmark = new CarDataSetBenchmark().run()

//  StdIn.readLine() // let it run until user presses return

  bindingFuture
    .flatMap(_.unbind()) // trigger unbinding from the port
    .onComplete(_ => system.terminate()) // and shutdown when done
}

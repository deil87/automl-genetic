package com.automl.route

import akka.actor.{Actor, ActorRef, ActorSystem, Cancellable, Props, Terminated}
import akka.http.scaladsl.model.ws.{Message, TextMessage}
import akka.http.scaladsl.server.Directives._
import akka.http.scaladsl.server.directives.ContentTypeResolver.Default
import akka.stream.OverflowStrategy
import akka.stream.scaladsl.{Flow, Sink, Source}

class WebSocketServiceRoute(notifier: ActorRef)(implicit as: ActorSystem) {

  def flow: Flow[Message, Message, Any] = {
    val sink = Sink.actorRef(notifier, 'sinkclose)
    val source = Source.actorRef(8, OverflowStrategy.fail).mapMaterializedValue { webClientActor ⇒
      notifier ! ('income → webClientActor)
      println(s"New client connected ${webClientActor.path.name}")
      webClientActor
    }
    Flow.fromSinkAndSource(sink, source)
  }

  val route = path("ws"){
    println("Hello ws")
    handleWebSocketMessages(flow)
  }
}


class ClientConnectionActor extends Actor {
  var connection: Option[ActorRef] = None

  var scheduler:Option[Cancellable] = None

  val receive: Receive = {
    case ('income, a: ActorRef) ⇒ {
      connection = Some(a)
      import scala.concurrent.duration._
      implicit val ec = context.system.dispatcher
      scheduler = Some(context.system.scheduler.schedule(50 milliseconds, 5 seconds) {
        a ! TextMessage.Strict(System.currentTimeMillis.toString)
      })
      context.watch(a)
    }
    case Terminated(a) if connection.contains(a) ⇒ connection = None; context.stop(self)
    case 'sinkclose ⇒
      context.stop(self)

    case TextMessage.Strict(t) ⇒ connection.foreach(_ ! TextMessage.Strict(s"echo $t"))
    case _ ⇒ // ignore
  }

  override def postStop(): Unit = connection.foreach(context.stop)
}
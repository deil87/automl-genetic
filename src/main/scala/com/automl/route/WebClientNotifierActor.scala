package com.automl.route

import akka.actor.{Actor, ActorRef, Cancellable}
import akka.http.scaladsl.model.ws.TextMessage

case class UpdateWeb(message: String)
case class UpdateWebWithJson(json: String)

class WebClientNotifierActor extends Actor {

  var webClientActor: Option[ActorRef] = None

  var scheduler:Option[Cancellable] = None

  val receive: Receive = {
    case ('income, a: ActorRef) ⇒ {
      webClientActor = Some(a)
//      import scala.concurrent.duration._
//      implicit val ec = context.system.dispatcher
//      scheduler = Some(context.system.scheduler.schedule(50 milliseconds, 5 seconds) {
//        val time = System.currentTimeMillis.toString
//        webClientActor.foreach(wc => wc ! TextMessage.Strict("Message from Notifier:" + time))
//        println(s"Time: $time")
//      })
      context.watch(webClientActor.get)
    }
    case UpdateWeb(t) ⇒ webClientActor.foreach(_ ! TextMessage.Strict(s"Progress: $t"))
    case UpdateWebWithJson(json) ⇒ webClientActor.foreach(_ ! TextMessage.Strict(json)) // TODO maybe we can pass key of update separately from json
    case 'sinkclose ⇒
      context.stop(self)
  }
}
package com.automl.route

import akka.http.scaladsl.server.Directives._
import akka.http.scaladsl.server.directives.ContentTypeResolver.Default

class StaticResourcesRoute {

  val staticResources =
    (get & pathPrefix("automl")){
      (pathEndOrSingleSlash /*& redirectToTrailingSlashIfMissing(TemporaryRedirect)*/) {
        getFromResource("static/automl-app/dist/index.html")
      } ~  {
        getFromResourceDirectory("static/automl-app/dist")
      }
    }
}

package com.automl

import scala.concurrent.{ExecutionContext, Future, Promise}

object FutureCancellable {

  def apply[T](body: =>T)(onCancelCode: => Unit)(implicit @deprecatedName('execctx) executor: ExecutionContext): (() => Unit, Future[T]) = {
    val p = Promise[T]
    val first = Future firstCompletedOf Seq(p.future, Future{ body })
    val cancellation: () => Unit = {
      () =>
        first onFailure { case e => onCancelCode}
        p failure new Exception
    }
    (cancellation, first)
  }
}

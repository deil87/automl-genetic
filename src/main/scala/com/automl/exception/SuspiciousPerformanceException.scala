package com.automl.exception

case class SuspiciousPerformanceException(message: String) extends Exception{
  override def getMessage: String = message + s"(${super.getMessage})"
}

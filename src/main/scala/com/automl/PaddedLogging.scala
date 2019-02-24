package com.automl

import com.typesafe.scalalogging.LazyLogging
import scalaz._
import Scalaz._

trait PaddedLogging extends LazyLogging{

  def logPaddingSize: Int // Maybe it would be simpler to use just var logPaddingSize and override it directly

  var overrideTo: Int = 0

  def debug(msg: String): Unit = msg.split("\n").foreach(msgLine => logger.debug(" " * Math.max(logPaddingSize, overrideTo) + msgLine))

  def info(msg: String): Unit = msg.split("\n").foreach(msgLine => logger.info(" " * Math.max(logPaddingSize, overrideTo) + msgLine))

  //With magnet pattern we maybe can overload it with one name
  def debugImpl(msg: String)(implicit logPaddingSize: Int): Unit = logger.debug(" " * logPaddingSize + msg)

}

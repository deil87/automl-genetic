package com.automl

import com.typesafe.scalalogging.LazyLogging
import scalaz._
import Scalaz._

trait PaddedLogging extends LazyLogging{

  def logPaddingSize: Int

  var overrideTo: Int = 0

  def debug(msg: String): Unit = msg.split("\n").foreach(msgLine => logger.debug(" " * Math.max(logPaddingSize, overrideTo) + msgLine))

  def info(msg: String): Unit = msg.split("\n").foreach(msgLine => logger.info(" " * Math.max(logPaddingSize, overrideTo) + msgLine))

  def debugImpl(msg: String)(implicit logPaddingSize: Int): Unit = logger.debug(" " * logPaddingSize + msg)

//  def setLogPadding(size: Int): Unit = overrideTo = size
}

package com.automl

import com.typesafe.config.{Config, ConfigFactory}

import scala.collection.mutable

object ConfigProvider {

  val overrideConfigs: mutable.Buffer[Config] = mutable.Buffer.empty[Config]

  lazy val config: Config = overrideConfigs :+ ConfigFactory.load() reduce( (l, r) => l.withFallback(r))

  def addOverride(conf: Config): mutable.Seq[Config] = {
    overrideConfigs += conf
  }
}


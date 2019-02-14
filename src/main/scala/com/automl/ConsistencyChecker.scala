package com.automl

trait ConsistencyChecker {

  def consistencyCheck[T](block: => T): Unit = {
    if(ConfigProvider.config.getBoolean("consistencyCheck")) block
  }
}

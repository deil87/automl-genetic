package com.automl

import com.automl.spark.SparkSessionProvider
import com.typesafe.config.{Config, ConfigFactory}
import org.scalatest.{FunSuite, Matchers}

class ConfigProviderTest extends FunSuite with SparkSessionProvider with Matchers {

  test("we can reset overridings with `ConfigProvider.clearOverride`") {

    val testOverride: Config = ConfigFactory.parseString(
      """
        |evolution {
        |  hpGridSearch = true
        |}
      """.stripMargin)

    ConfigProvider.addOverride(testOverride)

    val config = ConfigProvider.config.getConfig("evolution")
    val performGridSearch = config.getBoolean("hpGridSearch")

    performGridSearch shouldBe true

    val testOverrideWithFalse: Config = ConfigFactory.parseString(
      """
        |evolution {
        |  hpGridSearch = false
        |}
      """.stripMargin)

    ConfigProvider.clearOverride.addOverride(testOverrideWithFalse)

    val performGridSearchOverriden = ConfigProvider.config.getConfig("evolution").getBoolean("hpGridSearch")

    performGridSearchOverriden shouldBe false

  }
}

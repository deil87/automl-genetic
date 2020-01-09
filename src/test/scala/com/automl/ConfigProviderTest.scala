package com.automl

import com.automl.spark.SparkSessionProvider
import com.typesafe.config.{Config, ConfigFactory}
import org.scalatest.{FunSuite, Matchers}

class ConfigProviderTest extends FunSuite with SparkSessionProvider with Matchers {

  test("we can reset overridings with `ConfigProvider.clearOverride`") {

    ConfigProvider.clearOverride.addOverride("""
            evolution {
              hpGridSearch = true
            }
      """)

    val config = ConfigProvider.config.getConfig("evolution")
    val performGridSearch = config.getBoolean("hpGridSearch")

    performGridSearch shouldBe true

    ConfigProvider.clearOverride.addOverride(
      """
        |evolution {
        |  hpGridSearch = false
        |}
      """)

    val performGridSearchOverriden = ConfigProvider.config.getConfig("evolution").getBoolean("hpGridSearch")

    performGridSearchOverriden shouldBe false

  }
}

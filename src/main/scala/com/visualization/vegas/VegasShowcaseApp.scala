package com.visualization.vegas

object VegasShowcaseApp extends App{

  import vegas._

  val plot = Vegas("Country Pop").
    withData(
      Seq(
        Map("country" -> "USA", "population" -> 314),
        Map("country" -> "UK", "population" -> 64),
        Map("country" -> "DK", "population" -> 80)
      )
    ).
    encodeX("country", Nom).
    encodeY("population", Quant).
    mark(Bar)

  plot.show
}

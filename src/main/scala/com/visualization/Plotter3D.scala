package com.visualization

import org.apache.spark.ml.feature.PCA
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.jzy3d.chart.ChartLauncher
import org.jzy3d.chart.factories.AWTChartComponentFactory
import org.jzy3d.maths.{Coord2d, Coord3d, Rectangle}
import org.jzy3d.plot3d.primitives.Scatter
import org.jzy3d.plot3d.rendering.canvas.Quality
import utils.LabeledVector
import org.apache.spark.ml.linalg.{DenseVector, Vector => MLVector}
import org.jzy3d.colors.Color

class Plotter3D {


  def plotWithPCA(df: DataFrame) {
    import df.sparkSession.implicits._

    val pca = new PCA().setK(2).setInputCol("features").setOutputCol("pcaFeatures").fit(df.select(/*"label", */"features")/*.as[LabeledVector].map(_.features)*/)

    val projected = pca.transform(df)

    val pcaProjection: DataFrame = projected.drop("features").withColumnRenamed("pcaFeatures", "features")

    pcaProjection.show(20, false)
    val coordinates = pcaProjection.as[LabeledVector].map(lv => {
      val toArray: Array[Double] = lv.features.toArray
      (toArray(0), toArray(1), lv.label)
    }).collect().map { case (x, y, z) => new Coord3d(x, y, 0) }


    val scatter = new Scatter(coordinates/*, Array(Color.GREEN, Color.BLUE, Color.RED)*/)
    scatter.setWidth(10)
    val chart = AWTChartComponentFactory.chart(Quality.Advanced, "awt") // "newt"
    chart.getScene.add(scatter)
    val rectangle: Rectangle = new Rectangle(200, 200, 600, 600)
    ChartLauncher.openChart(chart, rectangle, "Title for 3D plotter")
    val t = 42
  }

}

package com.visualization

import com.automl.spark.SparkSessionProvider
import org.apache.spark.ml.feature.VectorAssembler
import org.scalatest.{FunSuite, Ignore}
import com.test.TestBase

@Ignore
class Plotter3DTest extends FunSuite with SparkSessionProvider with TestBase{
  import ss.implicits._

  import utils.SparkMLUtils._
//  import ss.sqlContext.implicits._


  test("testPlotWithPCA") {

    val initialDataset = ss.sparkContext.parallelize(
      Array(
        (  0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0, 0.0 ),
        (  0.0 , 1.0 , 0.0 , 0.0 , 0.0 , 0.0, 0.0 ),
        (  0.0 , 2.0 , 1.0 , 3.0 , 0.0 , 4.0, 4.0 ),
        (  0.0 , 2.0 , 1.0 , 3.0 , 0.0 , 4.0, 4.0 ),
        (  0.0 , 2.0 , 1.0 , 3.0 , 0.0 , 4.0, 4.0 ),
        (  0.0 , 2.0 , 1.0 , 3.0 , 0.0 , 4.0, 4.0 ),
        (  0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0, 0.0 )
      )
    ).toDF("f1", "f2", "f3", "f4", "f5", "f6", "label")

    val assembler =  new VectorAssembler()
      .setInputCols(Array("f1", "f2", "f3", "f4", "f5", "f6"))
      .setOutputCol("features")
    val aggregated = assembler.transform(initialDataset).select("features", "label")

    new Plotter3D().plotWithPCA(aggregated)

  }

}
case class UnlabeledFeatures( features: Array[Double])


package utils

import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql._
import org.apache.spark.sql.types.{StringType, StructField}

import scala.util.control.NonFatal
case class FeatureCorrelationInfo(xFieldName: String,
                                  percentOfValidData: Double,
                                  corr: Double,
                                  corrSpearman: Double,
                                  errorMessage: String = "")

class CorrelationHelper(dataFrame: DataFrame, targetField: String)(implicit ss: SparkSession)  {

  import ss.implicits._

  private def computeCorrelation(xFieldStruct: StructField) = {
    import breeze.linalg._


    val xField = xFieldStruct.name
    val xFieldType = xFieldStruct.dataType

    val correlationInfo =
      if (xFieldType.isInstanceOf[StringType]) {
        val filtered = dataFrame.filter(dataFrame(xField) rlike "[^NA]")
        try {
          val _ = filtered.select(xField).collect().map(_.getString(0).toDouble)

          val xRdd = filtered.select($"$xField".cast("double")).rdd.map(_.getDouble(0))
          val x = new DenseVector(xRdd.collect())

          val yRdd = filtered.select($"$targetField".cast("double")).rdd.map(_.getDouble(0))

          val corr = Statistics.corr(xRdd, yRdd)
          val corrSpearman = Statistics.corr(xRdd, yRdd, "spearman")

          val base = (filtered.count().toDouble / dataFrame.count()) * 100

          FeatureCorrelationInfo(xField, base, corr, corrSpearman)
        } catch {
          case NonFatal(e) =>
            val causeStr = filtered.select(xField).first().getString(0)
            FeatureCorrelationInfo(xField, 0, 0, 0, s"Error occured while $xField (First element in column: $causeStr)")
        }
      }
      else {
        val xRdd = dataFrame.select($"$xField".cast("double")).rdd.map(_.getDouble(0))
        val x = new DenseVector(xRdd.collect())

        val yRdd = dataFrame.select($"$targetField".cast("double")).rdd.map(_.getDouble(0))

        val corr = Statistics.corr(xRdd, yRdd)
        val corrSpearman = Statistics.corr(xRdd, yRdd, "spearman")

        FeatureCorrelationInfo(xField, 100, corr, corrSpearman)
      }

    println(correlationInfo)

    correlationInfo

    /*val f = Figure()
    val p = f.subplot(0)
    p += scatter(x, y, _ => 0.1)

    val base = (filtered.count().toDouble / dataFrame.count() ) * 100
    p.xlabel = s"$xField axis  ( Correlation based on $base% of data: $corr )"
    p.ylabel = s"$yField axis"


    f.saveas(s"${xField}__$yField.png")*/
  }

  def saveAndPrintCorrelations(fields: Option[Array[StructField]] = None): Unit = {
    fields.getOrElse(dataFrame.drop("id").schema.fields).foreach(f => println(f.name))
    val corrs = dataFrame.drop("id").schema.fields.map { field => computeCorrelation(field) }

    ss.sparkContext.parallelize(corrs.sortBy(-_.corr)).toDF().coalesce(1)
      .write
      .format("com.databricks.spark.csv")
      .option("header", true)
      .save(s"column_${targetField}_pearson")

    ss.sparkContext.parallelize(corrs.sortBy(-_.corrSpearman)).toDF().coalesce(1)
      .write
      .format("com.databricks.spark.csv")
      .option("header", true)
      .save(s"column_${targetField}_spearman")

    corrs.foreach(fci => println(s"Correlation for ${fci.xFieldName}_<->_${targetField} based on ${fci.percentOfValidData}% " +
      s"of data: Pearson:" + fci.corr + "Spearman:" + fci.corrSpearman))
  }
}

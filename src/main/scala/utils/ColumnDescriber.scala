package utils

import com.automl.spark.SparkSessionProvider
import org.apache.spark.mllib.linalg.{Vector => MLLibVector}
import org.apache.spark.sql._

object ColumnDescriber extends SparkSessionProvider{

  import ss.implicits._

  def describe(label: String)(df: DataFrame, column: String) = {
    println("/******** DESCRIBE " + label + "******************/")
    val columnSet: DataFrame = df.select($"$column").cache()
    columnSet.printSchema()
    println("Total count: " + columnSet.count())
    println("Is NULL count: " + columnSet.filter($"$column".isNull).count())
    println("NA count: " + columnSet.filter($"$column" rlike "NA").count())
    println("^NA count: " + columnSet.filter($"$column" rlike "[^NA]").count())

  }

}


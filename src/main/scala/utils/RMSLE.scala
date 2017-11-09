package utils

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object RMSLE {

  /*
  Compute based on 'prediction' and 'label' columns
   */
  def compute(observations: DataFrame)(implicit ss: SparkSession):Double = {
    import org.apache.spark.sql.functions._
    import ss.implicits._

    val sum = observations.withColumn("rmlse", pow(log($"prediction" + 1) - log($"label" + 1), 2))
    sum.agg(pow(mean("rmlse"), 0.5)).first().getDouble(0)
  }

}

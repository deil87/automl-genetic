package utils

import java.nio.file.Paths

import com.automl.classifier.ensemble.stacking.DNNIteratorParams
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql._
import org.apache.spark.ml.feature.{LabeledPoint => MLLabeledPoint, _}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import scala.collection.mutable.ArrayBuffer

object SparkMLUtils {

  implicit class LabelPointConverter(val lp: LabeledPoint) extends AnyVal {

    def toMLLabelPoint = MLLabeledPoint(lp.label, new DenseVector(lp.features.toArray))
  }

  implicit def datasetRowHelper(ds: Dataset[Row]) = new {
    import ds.sparkSession.implicits._
    def toMLLabelPoint: Dataset[MLLabeledPoint] = ds.as[MLLabeledPoint]
    def toMLLibLabelPoint: Dataset[LabeledPoint] = ds.as[MLLabeledPoint].map(mllp => LabeledPoint(mllp.label, Vectors.fromML(mllp.features)))
  }

  implicit class VectorsConverter(val vector: org.apache.spark.mllib.linalg.Vector) extends AnyVal {
    def toMLVector = new DenseVector(vector.toArray)
  }
  implicit class VectorsMLConverter(val vector: org.apache.spark.ml.linalg.Vector) extends AnyVal {
    def toMLLibVector = Vectors.dense(vector.toArray)
  }

  def loadDataset[SchemaType](resourcePath: String)(implicit ev: Encoder[SchemaType], ss: SparkSession) = {
    import ss.implicits._
    loadResourceDF(resourcePath).as[SchemaType]
  }

  def loadResourceDF(resourcePath: String)(implicit ss: SparkSession): DataFrame = {
    if(resourcePath.contains(".csv") )
      loadAbsoluteDF(resource(resourcePath))
    else if (resourcePath.contains(".json"))
      loadAbsoluteJSONDF(resource(resourcePath))
    else {
      throw new IllegalArgumentException("We can load only .csv and .json files for now")
    }
  }


  def loadAbsoluteDF(absolutePath: String)(implicit ss: SparkSession): DataFrame = {
    import ss.implicits._
    ss.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(absolutePath)
  }

  def loadAbsoluteJSONDF(absolutePath: String)(implicit ss: SparkSession): DataFrame = {
    import ss.implicits._
    ss.read
//      .option("multiline", "true")
      .json(absolutePath)
  }

  def loadParquetFromResources(resourcePath: String)(implicit ss: SparkSession): DataFrame = {
    loadParquet(resource(resourcePath))
  }

  def loadParquet(absolutePath: String)(implicit ss: SparkSession): DataFrame = {
    ss.read.parquet(absolutePath)
  }

  def resource(path: String): String = {
    val absolutePath = Paths.get(getClass.getResource(path).toURI).toString
    println(absolutePath)
    absolutePath
  }

  implicit class DataFrameDNNHelper(val df: sql.DataFrame) extends AnyVal {

    import df.sparkSession.implicits._
    def toDataSetIterator(dnnParams: DNNIteratorParams) = {
      DeepLearningHelper.trainingIteratorFromDF(df, dnnParams)
    }

  }

  implicit class DataFrameHelper(val df: sql.DataFrame) extends AnyVal {

    import df.sparkSession.implicits._

    def applyIndexer(indexer: (Dataset[_]) => StringIndexerModel) =  {
      indexer(df).transform(df)
    }

    def applyIndexer(indexer: StringIndexer) =  {
      indexer.fit(df).transform(df)
    }

    def applyIndexersWithReplacing(indexers: Seq[StringIndexer]) =  {
      indexers.foldLeft(df){(res, indexer) =>
        indexer.fit(res)
          .transform(res)
          .drop(indexer.getInputCol)
          .withColumnReplace(indexer.getOutputCol, indexer.getInputCol)
//          .showN_AndContinue(20)
      }
    }

    def applyBucketizer(bucketizer:  Bucketizer) = bucketizer.transform(df)

//    def applyIndexers(indexers: ((Dataset[_]) => StringIndexerModel)*) =  {
////      val indexersSeq: Seq[(Dataset[_]) => StringIndexerModel] = indexers:_*
//      indexers.foldLeft(df)( (res, next) => next(df).transform(res))
//    }

    def applyDFTransformation(tr: sql.DataFrame => sql.DataFrame) = {
      tr(df)
    }

    def applyTransformation(tr: {def transform(ds: Dataset[_]): sql.DataFrame}): sql.DataFrame = {
      tr.transform(df)
    }

    def applyTransformation(scaler: StandardScaler): sql.DataFrame = {
      val scalerModel = scaler.fit(df)
      scalerModel.transform(df)
    }

    def applyTransformation(scaler: MinMaxScaler): sql.DataFrame = {
      val scalerModel = scaler.fit(df)
      scalerModel.transform(df)
    }

    def applyVectorAssembler(fun: Array[String] => Transformer, exclude: Array[String]): sql.DataFrame = {
      val scaler: Transformer = fun(df.schema.names.diff(exclude))
      scaler.transform(df)
    }

    def applyPCATransformation(pca: PCA): sql.DataFrame = {
      val pcaModel = pca.fit(df)
      pcaModel.transform(df)
    }


    def indexed(sourceFeatureName: String): DataFrame = {
      val indexer = new StringIndexer().setInputCol(sourceFeatureName).setOutputCol(sourceFeatureName + "_i").fit(df)
      indexer.transform(df)
    }
    def indexedSeq(featuresNames: Set[String]) = {
      featuresNames.foldLeft(df)((res, next) => res.indexed(next))
    }


    def ohEncoded(sourceFeatureName: String): DataFrame = {
      val encoder = new OneHotEncoder()
        .setInputCol(sourceFeatureName)
        .setOutputCol(sourceFeatureName + "_ohe")
      encoder.transform(df)
    }

    def indexedAndOHE(sourceFeatureName: String): DataFrame = {
      val indexer = new StringIndexer().setInputCol(sourceFeatureName).setOutputCol(sourceFeatureName + "_i").fit(df)
      val indexed = indexer.transform(df)
      val encoder = new OneHotEncoder()
        .setDropLast(false)
        .setInputCol(sourceFeatureName+ "_i")
        .setOutputCol(sourceFeatureName + "_ohe")
      encoder.transform(indexed)
    }

    def ohEncodedSet(featuresNames: Set[String]) = {
      featuresNames.foldLeft(df)((res, next) => res.ohEncoded(next))
    }

    def indexedAndOHESet(featuresNames: Set[String]) = {
      featuresNames.foldLeft(df)((res, next) => res.indexedAndOHE(next))
    }

    def repairNullWithRangeMean(field: String, start: Int, end: Int = 0) = {
      checkExistenceAndType(field, DoubleType)
      val window = Window.orderBy("timestamp").rowsBetween(start, end)

      df.withColumnReplace(s"$field", when($"$field".isNull, mean(field).over(window)).otherwise($"$field").cast("double"))
    }

    def repairNullWithRangeMeanRate(field: String, start: Int) = {
      checkExistenceAndType(field, DoubleType)
      val window = Window.orderBy("timestamp").rowsBetween(start, 1)
      val lagWindow = Window.orderBy("timestamp")
      df
        .withColumn(s"$field-lag", coalesce(lag(field, 1).over(lagWindow), $"$field", lit(0)))
//        .withColumn(s"$field-mean-win", mean($"$field").over(window))
//        .withColumn(s"$field-rate-win", mean($"$field" - $"$field-lag").over(window))
        .withColumnReplace(s"$field",
          when($"$field".isNull, coalesce(mean($"$field").over(window) + mean($"$field" - $"$field-lag").over(window), lit(0.0)))
            .otherwise($"$field").cast("double"))
    }

    def repairNAorNullWithMean(field: String)(implicit sparkSession: SparkSession) = {
      import sparkSession.implicits._
      df.withColumnReplace(field, when(($"$field" rlike("NA")) || $"$field".isNull, df.select(mean($"$field")).first()(0)).otherwise($"$field").cast("double"))
    }

    def repairNAWithMedian(field: String)(implicit sparkSession: SparkSession) = {
      import sparkSession.implicits._
      val allNonMissing = df.filter((df(field) rlike ("[^NA]")) && df(field).isNotNull)
      df.withColumnReplace(field, when(($"$field" rlike("NA")) || $"$field".isNull, median(allNonMissing, field)).otherwise($"$field").cast("double"))
    }

    def repairNAWithMedianRound(field: String)(implicit sparkSession: SparkSession) = {
      import sparkSession.implicits._
      val allNonMissing = df.filter((df(field) rlike ("[^NA]")) && df(field).isNotNull)
      df.withColumnReplace(field, when(($"$field" rlike("NA")) || $"$field".isNull, round(lit(median(allNonMissing, field)))).otherwise($"$field").cast("double"))
    }

    def repairZeroWithMedian(field: String)(implicit sparkSession: SparkSession) = {
      import sparkSession.implicits._
      val allNonZero = df.filter(df(field) =!= 0)
      df.withColumnReplace(field, when($"$field" === 0.0, median(allNonZero, field)).otherwise($"$field").cast("double"))
    }

    private def median(list: DataFrame,field: String): Double = {
      val sorted = list.select(df(field).cast(DoubleType)).sort(df(field))
      val count = sorted.count().toInt // dangerous toInt
      val median =
        if (count % 2 == 1) {
          sorted.take((count / 2) + 1).last.getDouble(0)
        }
        else {
          sorted.take((count / 2) + 1).takeRight(2).map(_.getDouble(0)).sum / 2
        }
      median
    }

    def repairZeroWithMean(field: String)(implicit sparkSession: SparkSession) = {
      import sparkSession.implicits._
      df.withColumnReplace(field, when($"$field" === 0, df.filter($"$field" =!= 0).select(mean($"$field")).first()(0)).otherwise($"$field").cast("double"))
    }

    def repairNAWithZero(field: String)(implicit sparkSession: SparkSession) = {
      import sparkSession.implicits._
      df.withColumnReplace(field, when(($"$field" rlike("NA")) || $"$field".isNull, 0).otherwise($"$field").cast("int"))
    }

    def repairWithMax(field: String)(implicit sparkSession: SparkSession) = {
      import sparkSession.implicits._
      df.withColumnReplace(field, when($"$field" rlike("NA"), df.select(max($"$field")).first()(0)).otherwise($"$field").cast("int"))
    }

    def repairNullWithMax(field: String)(implicit sparkSession: SparkSession) = {
      import sparkSession.implicits._
      df.withColumnReplace(field, when($"$field".isNull, df.select(max($"$field")).first()(0)).otherwise($"$field").cast("int"))
    }

    def checkZeroValues(field: String)(implicit sparkSession: SparkSession) = {
      import sparkSession.implicits._
      val count: Long = df.filter($"$field" === 0).count()
      require(count == 0, s"Check zeroes failed ($field): $count")
      df
    }

    def checkNullValues(field: String) = {
      val count: Long = df.filter($"$field".isNull).count()
      require(count == 0, s"Check NULL values failed ($field): $count")
      df
    }
    def checkNAValues(field: String) = {
      val count: Long = df.filter($"$field"  rlike("NA")).count()
      require(count == 0, s"Check NA values failed ($field): $count")
      df
    }

    def repairWithMeanSet(fields: Set[String])(implicit sparkSession: SparkSession) = {
      fields.foldLeft(df)((res, next) => res.repairNAorNullWithMean(next)(sparkSession))
    }

    def toInt(field: String)(implicit sparkSession: SparkSession) = {
      import sparkSession.implicits._
      df.withColumnReplace(field, $"$field".cast("int"))
    }
    def toDouble(field: String) = {
      df.withColumnReplace(field, $"$field".cast("double"))
    }
    def toLong(field: String) = {
      df.withColumnReplace(field, $"$field".cast("long"))
    }
    def toDouble(field: Iterable[String])(implicit sparkSession: SparkSession) = {
      import sparkSession.implicits._
      field.foldLeft(df)((res, next) => res.withColumnReplace(next, $"$next".cast("double")))
    }

      def castColumnTo( cn: String, tpe: DataType ) : sql.DataFrame = {
      df.withColumn( cn+"_casted", df(cn).cast(tpe) ).drop(cn).withColumnRenamed(cn+"_casted", cn)
    }

    def withColumnReplace(colName: String, col: Column) = {
      df.withColumn(colName+"_new", col).drop(colName).withColumnRenamed(colName+"_new", colName)
    }

    def withColumnReplace(toOverride: String, overrideBy: String) = {
      df.drop(toOverride).withColumnRenamed(overrideBy, toOverride)
    }

    def showCount(prefix: String = "Count") = println(prefix + "_" + df.count())

    def showAll() = df.show(df.count().toInt, false)
    def showAllAndContinue = {
      df.show(df.count().toInt, false)
      df
    }
    def showFieldAndContinue(field: String) = {
      df.select(field).show(df.count().toInt, false)
      df
    }
    def showN_AndContinue(n: Int, label: String = "") = {
      println(s"Show $n rows (label:" + label + ")")
      df.show(n, false)
      df
    }

    def showCount_AndContinue(label: String = "") = {
      println(s"Count# " + label + ": " + df.count())
      df
    }

    def sampleRand(number: Int, seed: Long) = {
      df.orderBy(rand(seed)).limit(number)
    }

    def printSchema_AndContinue = {
       df.printSchema(); df.schema.fields.find(field => field.name == "label").map(_.metadata).foreach(println); df
    }

    def showID_AndContinue(id: Int) = {
      df.where(df("id") === id).show(1, false)
      df
    }

    def filterOutNA(featureName: String) = {
      checkExistenceAndType(featureName, StringType)
      df.filter( df(featureName) rlike "[^NA]")
    }

    def filterOutZeroAndNull(featureName: String, columnType: NumericType = DoubleType ) = {
      checkExistenceAndType(featureName, columnType)
      df.filter((df(featureName) =!= 0) && df(featureName).isNotNull)
    }

    def filterOutNull(featureName: String, columnType: NumericType = DoubleType ) = {
      checkExistenceAndType(featureName, columnType)
      df.filter(df(featureName).isNotNull)
    }
    def filterNA(featureName: String) = {
      df.filter( (df(featureName) rlike "NA") || df(featureName).isNull)
    }

    def filterNull(featureName: String, columnType: NumericType = DoubleType) = {
      checkExistenceAndType(featureName, columnType)
      df.filter(df(featureName).isNull)
    }

    def filterZeroAndNull(featureName: String) = {
      checkExistenceAndType(featureName, DoubleType)
      df.filter((df(featureName) === 0) || df(featureName).isNull)
    }

    def toTrainTestPairs(nFolds:Int, seed: Long) = {
      val splitsRdd = MLUtils.kFold(df.rdd, nFolds, seed )
      val splits = splitsRdd.map { case (training, validation) =>
        val validationCount = validation.count()
        val trainCount = training.count()
        val trainingSplitDF = df.sparkSession.createDataFrame(training, df.schema).cache()
        val validationSplitDF = df.sparkSession.createDataFrame(validation, df.schema).cache()
        require(validationCount > 0 && trainCount > 0, "Validation or training split is of 0 size")
        (trainingSplitDF, validationSplitDF)
      }
      splits
    }

    def checkTrainCount = {
      require(df.count() == 30471); df
    }
    def checkTestCount = {
      require(df.count() == 7662); df
    }
    def checkTotalCount = {
      require(df.count() == 38133); df
    }

    private def checkExistenceAndType(featureName: String, columnType: NumericType): Unit = {
      val field = df.schema.fields.find(_.name == featureName)
      require(field.isDefined, s"Field `$featureName` is absent .")
      require(field.get.dataType == columnType, s"Wrong type of field `$featureName`. Should be NumericType but ${field.get.dataType.typeName}")
    }

    private def checkExistenceAndType(featureName: String, columnType: StringType): Unit = {
      val field = df.schema.fields.find(_.name == featureName)
      require(field.isDefined && field.get.dataType == columnType, "Field is absent or of a wrong type.")
    }

    def computeColumnNamesForFeaturesVector(features: Array[String]): Array[String] = {
      val firstRow = df.select(features.head, features.tail: _*).first()

      firstRow.schema.fields.foldLeft(new ArrayBuffer[String]()) { (arr, strField) =>
        val span = firstRow.getValuesMap(Seq(strField.name)).head._2.asInstanceOf[Any] match {
          case x: SparseVector => x.asInstanceOf[SparseVector].size
          case x => 1
        }
        for (i <- 0 until span) {
          arr += strField.name + (if (span == 1) "" else i)
        }
        arr
      }.toArray
    }
  }

}

case class LabeledFeatures(label: Double, features: Array[Double]) {
  def toLabeledPoints = LabeledPoint(label, Vectors.dense(features))
  def toMLLabeledPoints =  MLLabeledPoint(label, new DenseVector(features))
}

final case class LabeledVector(label: Double, features: org.apache.spark.ml.linalg.Vector) {
  def toLibLabeledPoints =  LabeledPoint(label, Vectors.dense(features.toArray))
}

final case class UnlabeledVector(id: Double, features: org.apache.spark.ml.linalg.Vector)


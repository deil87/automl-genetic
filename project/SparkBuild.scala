import sbt.Keys._
import sbt._

object SparkBuild extends Build {
  lazy val testLibDependencies = List(
    "org.scalacheck" %% "scalacheck" % "1.13.4" % "test",
    "org.scalactic" %% "scalactic" % "3.0.1",
    "org.scalatest" %% "scalatest" % "3.0.1" % "test",
    "com.storm-enroute" %% "scalameter" % "0.7"
  )
  lazy val loggingLibDependencies = List()

  object Version {
    val deeplearning4jVersion = "0.9.1"
  }

  lazy val sparkDependencies = List(
    "org.apache.spark" %% "spark-core" % "2.1.0",
    "org.apache.spark" %% "spark-sql" % "2.1.0",
    "org.apache.spark" %% "spark-streaming" % "2.1.0",
    "org.apache.spark" %% "spark-mllib" % "2.1.0",
    "org.apache.spark" % "spark-streaming-twitter_2.11" % "1.6.3",
    "org.scalanlp" %% "breeze" % "0.13",
    "org.scalanlp" %% "breeze-viz" % "0.13" withSources(),
    "com.datastax.spark" % "spark-cassandra-connector_2.11" % "2.0.0-M3",
    "org.apache.spark" % "spark-streaming-kafka_2.11" % "1.6.3",
    "org.scalafx" %% "scalafx" % "2.2.76-R11",
    "com.teamdev.jxbrowser" % "jxbrowser-mac" % "6.0",


    "org.bytedeco" % "javacpp" % "1.3.3",
    "org.nd4j"% "nd4j-native" % Version.deeplearning4jVersion,
    "org.nd4j"% "nd4j-native-platform" % Version.deeplearning4jVersion,
    "org.deeplearning4j" % "deeplearning4j-core" % Version.deeplearning4jVersion,
    "org.deeplearning4j" % "deeplearning4j-nn" % Version.deeplearning4jVersion,
    "org.deeplearning4j" % "deeplearning4j-ui-components" % Version.deeplearning4jVersion ,

    //h2o
    "ai.h2o" % "sparkling-water-core_2.11" % "2.2.0",

    "nz.ac.waikato.cms.weka" % "weka-stable" % "3.8.1",
    "nz.ac.waikato.cms.weka" % "SMOTE" % "1.0.3",

    "org.jzy3d" % "jzy3d-api" % "1.0.0",

    "io.kamon" %% "kamon-core" % "1.0.0",
    "io.kamon" %% "kamon-prometheus" % "1.0.0",
    "io.kamon" %% "kamon-datadog" % "1.0.0-RC1-2dcf4510efe9df12e640504bfa30ecabb3422638",
//    "io.kamon" %% "kamino-reporter" % "1.0.0",
//    "io.kamon" %% "kamon-datadog" % "0.6.8",
//      "io.kamon" %% "kamon-core" % "0.6.7",
//      "io.kamon" %% "kamon-datadog" % "0.6.7",

    "io.spray" %% "spray-can" % "1.3.4",
    "io.spray" %% "spray-client" % "1.3.4",
    "io.spray" %%  "spray-json" % "1.3.3",
    "net.ruippeixotog" %% "scala-scraper" % "2.0.0-RC2",
    "com.typesafe.slick" %% "slick" % "3.2.0",

    "ml.dmlc" % "xgboost4j" % "0.7",
    "ml.dmlc" % "xgboost4j-spark" % "0.7"
  )


  lazy val libDependencies = sparkDependencies ++ testLibDependencies ++ loggingLibDependencies

  val buildSettings = Seq(
    version := "1.0",
    organization := "com.example",
    scalaVersion in ThisBuild := "2.11.7",
    resolvers ++= Nil,
    javaOptions += "-Xmx3000m"
  )

  lazy val root: Project = Project(
    "root",
    file("."),
    settings = buildSettings ++ Seq(
      libraryDependencies ++= libDependencies,
      resolvers += "com.teamdev" at "http://maven.teamdev.com/repository/products",
      resolvers += "jzy3d-snapshots" at "http://maven.jzy3d.org/releases",
      resolvers += "Local Maven Repository" at "file://"+ Path.userHome+"/.m2/repository",
      resolvers += Resolver.bintrayRepo("kamon-io", "snapshots"),
      ivyScala := ivyScala.value map { _.copy(overrideScalaVersion = true) }
      )

  )

}

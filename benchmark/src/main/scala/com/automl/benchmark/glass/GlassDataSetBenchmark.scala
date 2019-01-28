package com.automl.benchmark.glass

import akka.actor.ActorSystem
import com.automl.evolution.dimension.{TemplateEvolutionDimension, TemplateHyperParametersEvolutionDimension}
import com.automl.spark.SparkSessionProvider
import com.automl.template.LeafTemplate
import com.automl.template.simple._
import com.automl.{AutoML, TPopulation}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.rand
import org.bytedeco.javacpp.opencv_ml.LogisticRegression
import utils.SparkMLUtils

class GlassDataSetBenchmark(implicit as: ActorSystem) extends SparkSessionProvider{

  import utils.SparkMLUtils._

  def run() = {

    val seed: Seq[LeafTemplate[SimpleModelMember]] = Seq(
      LeafTemplate(new LogisticRegressionModel()),
      LeafTemplate(Bayesian()),
//      LeafTemplate(GradientBoosting()), //TODO multiclass classification case is not supported
      //        LeafTemplate(NeuralNetwork(Array(5,10,5))), // TODO need to implement detection of features number and number of classes
      LeafTemplate(DecisionTree())
    )

    val seedPopulation = new TPopulation(seed)

    val population = TPopulation.fromSeedPopulation(seedPopulation).withSize(10).build

    val glassDF = SparkMLUtils.loadResourceDF("/glass/glass.csv")

    val features = Array("RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe")

    val featuresColName: String = "features"

    def featuresAssembler = {
      new VectorAssembler()
        .setInputCols(features)
        .setOutputCol(featuresColName)
    }

    val preparedGlassDF = glassDF
      .orderBy(rand())  // Shuffling
      .applyTransformation(featuresAssembler)
      .toLong("Id")
      .withColumnRenamed("Id", "uniqueIdColumn")
      .withColumnRenamed("TypeOfGlass", "label")
      .showN_AndContinue(10)
      .cache()

    //Note we are passing whole dataset and inside it is being splitted as train/test. Maybe it is a good idea to hold test split for a final examination.
    val autoMl = new AutoML(
      data = preparedGlassDF,
      responseColumn = "label",
      maxTime = 2 * 60000,
      useMetaDB = false,
      initialPopulationSize = Some(7),
      seedPopulation = seedPopulation,
      maxGenerations = 5)

    val templateEvDim = new TemplateEvolutionDimension(2)(as)
    val hyperParamsEvDim = new TemplateHyperParametersEvolutionDimension(2)

    autoMl.runEvolution(templateEvDim, hyperParamsEvDim)
  }

}

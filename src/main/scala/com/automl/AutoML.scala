package com.automl

import akka.actor.ActorSystem
import com.automl.dataset._
import com.automl.evolution.dimension.TemplateEvolutionDimension
import com.automl.helper._
import com.automl.population.TPopulation
import com.automl.problemtype.ProblemType.BinaryClassificationProblem
import com.automl.problemtype.{ProblemType, ProblemTypeThresholdEstimator}
import com.automl.report.AutoMLReporter
import com.typesafe.scalalogging.LazyLogging
import kamon.Kamon
import org.apache.spark.sql.DataFrame

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.concurrent.{Await, Future, _}
import scala.util.Random

class MetaDB() {
  def getPopulationOfTemplates = ???  // Population of base models?
}

/**
  * Consider using Builder pattern for setting all the parameters.
  */
class AutoML(data: DataFrame,
             idColumn: Option[String] = None,
             responseColumn: String = null, // TODO move logic for preparation of the DataSet inside AutoML
             maxTime: Long,
             maxGenerations: Int = 10,
             maxEvolutions: Int = 5,
             maxDepthOfEnsemble: Int = 3,
             maxNumberOfChildrenPerEnsemblingNode: Int = 10,
             useMetaDB: Boolean,
             // TODO make it optional because of useMetaDB. Maybe we don't need this parameter as we should select appropriate models for a given problem type
             seedPopulation: Option[TPopulation] = None,
             initialPopulationSize: Option[Int] = None,
             isBigSizeThreshold: Long = 500,
             isBigDimensionsThreshold: Long = 200,
             initialSampleSize: Long = 500,
             dataSetSizeEvolutionStrategy: DataSetSizeEvolutionStrategy = new RandomDataSetSizeEvolutionStrategy(),
             seed: Long = new Random().nextLong()
            ) extends LazyLogging {

  implicit val logPaddingSize: Int = 0

  require(!useMetaDB && initialPopulationSize.isDefined, "If there is no metaDB information then we should start from scratch with population of defined size")

  require(data.columns.contains(responseColumn), s"Response column with name $responseColumn is not presented in the dataset")

  // Choose problem type estimator based on input parameter or config file? How to separate what goes where -> config/parameters?
  val problemType: ProblemType = new ProblemTypeThresholdEstimator(10).estimate(data, responseColumn)

  lazy val totalDataSize: Long = getDataSize(data)

  val isDataBig: Boolean = isDataBig(data)

  lazy val evolutionDataSizeFactor: Long = Math.max(totalDataSize / maxEvolutions, 500)

  def calculateTimeBoxes(isDataBig: Boolean): EvolutionTimeBoxes = {
    val allowanceOfEvolutions = if(isDataBig) maxEvolutions else 1
    val strategy: EqualEvolutionsStrategy = EqualEvolutionsStrategy(maxTime, allowanceOfEvolutions)
    val builder = EvolutionTimeBoxesBuilder(maxTime, allowanceOfEvolutions).withSplittingStrategy(strategy)
    builder.build
  }

  // <editor-fold defaultstate="collapsed" desc="Kamon metrics">

  val evolutionNumberKamon = Kamon.gauge("kamon.automl.evolution_number")
  val generationNumberKamon = Kamon.gauge("kamon.automl.generation_number")
  val currentDatasizeKamon = Kamon.gauge("kamon.automl.current_datasize")
  // </editor-fold>

  def isDataBig(df: DataFrame): Boolean = {
    def numberOfDimensions: Int = df.columns.length
    numberOfDimensions >= 200 || totalDataSize >= isBigSizeThreshold
  }

  def getDataSize(df: DataFrame): Long = df.count()

  def isDataSetBalanced: Boolean = problemType match {
    case BinaryClassificationProblem =>
      val (_,counts) = data.select("indexedLabel").rdd.map(value => value.getDouble(0)).histogram(2)
      val sortedCounts = counts.sortWith(_ < _)
      if(sortedCounts(0).toDouble / sortedCounts(1) < 0.8) false else true
    case _ => true
  }

  implicit val samplingStrategy: SamplingStrategy = if(isDataSetBalanced) new RandomSampling() else new StratifiedSampling()

  val metaDB = new MetaDB() // TODO How it should look like?

  /*Probably we need a tree of dimensions in order to predefine dependencies*/
  def runEvolution(implicit as: ActorSystem): Unit = {

    // TODO instead of using Option for any disabled dimension we should use Map with keys( dimensionName as a key) to be able to depend on arbitrary number of coevolutions.
    val templateEvDim = new TemplateEvolutionDimension(initialPopulation = seedPopulation, problemType = problemType)(as, logPaddingSize + 0)

    //TODO implement changing of dataset's size as SizeEvolution dimension.
    var workingDataSet: DataFrame = if(isDataBig) {
      val initialSamplingRatio = initialSampleSize.toDouble / totalDataSize.toDouble
      //TODO maybe we can start using EvolutionStrategy even here?
      samplingStrategy.sampleExact(data, initialSamplingRatio, seed).cache()
    } else data


    var currentDataSize = workingDataSet.count()
    currentDatasizeKamon.set(currentDataSize)

    //The fitness of each template is updated during evolutions and when the optimization terminates,
    // winning templates are saved as a new record into the metadatabase or corresponding records are
    // updated with the new templates.
    var evolutionNumber = 0
    evolutionNumberKamon.set(0)

    val startTime = System.currentTimeMillis()

    val timeBoxes = calculateTimeBoxes(isDataBig)
    logger.info("TimeBoxes schedule " + timeBoxes.timeBoxes.map(_.upperBoundary).mkString(","))

    //TODO Consider not creating timeboxes when dataset is not big. We can use generations only as we will not be increasing dataset size over evolutions.
    timeBoxes.timeBoxes foreach { timeBox =>
      // Following should be equal to current (timebox.limit - previousTimeBox.limit)
      def restOfTheTimeBox = Math.max(timeBox.upperBoundary - (System.currentTimeMillis() - startTime), 1000)

      val (_,counts) = workingDataSet.select("indexedLabel").rdd.map(value => value.getDouble(0)).histogram(2)
      logger.debug(s"Distribution of classes for classification after sampling: ${counts.mkString(" , ")} ")

      val timeBoxCalculations = Future {
        logger.info(s"$timeBox launched:")
        currentDataSize = workingDataSet.count()
        logger.info(s"Evolution number $evolutionNumber is launched with datasize = $currentDataSize (rows) out of $totalDataSize (rows) ...")

        def condition = System.currentTimeMillis() - startTime < timeBox.upperBoundary

        var generationNumber = 0
        generationNumberKamon.set(0)

        while (condition) {

          logger.info(s"Time left: ${(maxTime - System.currentTimeMillis() + startTime) / 1000}")
          logger.info(s"Generation number $generationNumber is launched ( evolution number $evolutionNumber)")

          templateEvDim.evolveFromLastPopulation(workingDataSet)

          generationNumber += 1
          generationNumberKamon.increment(1)
        }
      }

      try {
        Await.result(timeBoxCalculations, restOfTheTimeBox.milliseconds)
      } catch {
        case ex:TimeoutException =>
          val infoMessage = s"Timeout for $timeBox has happened."
          logger.info(infoMessage)
          logger.debug(infoMessage + ex.getMessage)
          if (currentDataSize < totalDataSize) {
            val newDataSize = currentDataSize + evolutionDataSizeFactor
            workingDataSet = dataSetSizeEvolutionStrategy.evolve(workingDataSet, newSize = newDataSize, maxEvolutions, data, seed)
            evolutionNumber += 1
          }
      }
    }

    AutoMLReporter.show(templateEvDim.hallOfFame, problemType) // TODO we need to return best individual for validation on testSplit
  }

}



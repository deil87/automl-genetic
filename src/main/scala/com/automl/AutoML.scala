package com.automl


import akka.actor.{ActorSelection, ActorSystem}
import akka.util.Timeout
import com.automl.dataset._
import com.automl.evolution.dimension.TemplateEvolutionDimension
import com.automl.evolution.evaluation.EvaluationContextInfo
import com.automl.helper._
import com.automl.population.TPopulation
import com.automl.problemtype.ProblemType.BinaryClassificationProblem
import com.automl.problemtype.{ProblemType, ProblemTypeThresholdEstimator}
import com.automl.report.AutoMLReporter
import com.automl.route.UpdateWebWithJson
import com.typesafe.scalalogging.LazyLogging
import kamon.Kamon
import org.apache.spark.sql.DataFrame
import spray.json.DefaultJsonProtocol

import scala.collection.mutable
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.concurrent.{Await, Future, _}
import scala.util.Random

class MetaDB() {
  def getPopulationOfTemplates = ???  // Population of base models?
}

case class EvolutionProgressDTO( evolution: Int, generation: Int, timeBox: String, stage: String, key: String = "evolutionProgress")

object EvaluatedTemplateDataDTOJsonProtocol extends DefaultJsonProtocol {
  implicit val format = jsonFormat5(EvolutionProgressDTO.apply)
}

/**
  * Consider using Builder pattern for setting all the parameters.
  */
class AutoML(data: DataFrame,
             idColumn: Option[String] = None,
             responseColumn: String = null, // TODO move logic for preparation of the DataSet inside AutoML
             maxTime: Long,
             maxEvolutions: Int = 5,
             maxGenerations: Int = 10,
             maxDepthOfEnsemble: Int = 3,
             maxNumberOfChildrenPerEnsemblingNode: Int = 10,
             useMetaDB: Boolean,
             // TODO make it optional because of useMetaDB. Maybe we don't need this parameter as we should select appropriate models for a given problem type
             seedPopulation: Option[TPopulation] = None,
             initialPopulationSize: Option[Int] = None,
             isBigSizeThreshold: Long = 500,
             isBigDimensionsThreshold: Long = 200,
             initialSampleSize: Long = 500, // Maybe maxEvolutions is enough to estimate initial sample size?
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

  lazy val evolutionDataSizeIncrement: Long = (totalDataSize - initialSampleSize) / (maxEvolutions - 1) // Start with `initialSampleSize` for the first evolution then for N-1 evolutions will be increasing equally

  def calculateTimeBoxes(isDataBig: Boolean): EvolutionTimeBoxes = {
    val allowanceOfEvolutions = if(isDataBig) maxEvolutions else {
      logger.info("Data was considered as small therefore `maxEvolutions` was set to 1")
      1
    }
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
    totalDataSize >= isBigSizeThreshold || numberOfDimensions >= 200
  }

  def getDataSize(df: DataFrame): Long = df.count()

  def isDataSetBalanced: Boolean = problemType match {
    case BinaryClassificationProblem =>
      val (_,counts) = data.select("indexedLabel").rdd.map(value => value.getDouble(0)).histogram(2)
      val sortedCounts = counts.sortWith(_ < _)
      if(sortedCounts(0).toDouble / sortedCounts(1) < 0.8) false else true
    case _ => true
  }

  implicit val samplingStrategy: SamplingStrategy = if(isDataSetBalanced) new RandomRowsSampling() else new StratifiedRowsSampling()

  val metaDB = new MetaDB() // TODO How it should look like?

  /*Probably we need a tree of dimensions in order to predefine dependencies*/
  def runEvolution(implicit as: ActorSystem): mutable.PriorityQueue[EvaluatedTemplateData] = {

    // Updating UI
//    import scala.concurrent.ExecutionContext.Implicits.global
    val webClientNotifier: ActorSelection = as.actorSelection("user/webClientNotifier")

    // TODO instead of using Option for any disabled dimension we should use Map with keys( dimensionName as a key) to be able to depend on arbitrary number of coevolutions.
    val templateEvDim = new TemplateEvolutionDimension(initialPopulation = seedPopulation, problemType = problemType)(as, logPaddingSize + 0)

    var workingDataSet: DataFrame = if(isDataBig) {
      val initialSamplingRatio = initialSampleSize.toDouble / totalDataSize.toDouble
      samplingStrategy.sampleRatio(data, initialSamplingRatio, seed).cache()
    } else data


    var currentDataSize = workingDataSet.count()
    currentDatasizeKamon.set(currentDataSize)

    //The fitness of each template is updated during evolutions and when the optimization terminates,
    // winning templates are saved as a new record into the metadatabase or corresponding records are
    // updated with the new templates.
    var evolutionNumber = 1
    evolutionNumberKamon.set(0)

    val startTime = System.currentTimeMillis()

    val timeBoxes = calculateTimeBoxes(isDataBig)
    logger.info("TimeBoxes schedule: " + timeBoxes.timeBoxes.map(_.upperBoundary).mkString(","))

    def totalTimeLeft = (maxTime - System.currentTimeMillis() + startTime) / 1000

    //TODO Consider not creating timeboxes when dataset is not big. We can use generations only as we will not be increasing dataset size over evolutions.
    timeBoxes.timeBoxes foreach { timeBox =>

      // Following should be equal to current (timebox.limit - previousTimeBox.limit)
      def restOfTheTimeBox = Math.max(timeBox.upperBoundary - (System.currentTimeMillis() - startTime), 1000)

      if(totalTimeLeft < 10) {
        // TODO be smarter and add this time to previous timeBox. Or just split better from the beginning
        logger.info(s"$timeBox is skipped as it is less than 10 secs left.")
      } else {

        val presentLabels = workingDataSet.select("indexedLabel").distinct()
        presentLabels.show(false)
        val numberOfClasses = presentLabels.count().toInt
        workingDataSet.show(false)
        val (buckets, counts) = workingDataSet.select("indexedLabel").rdd.map(value => value.getDouble(0)).histogram(numberOfClasses + 1)
        logger.debug("Buckets:" + buckets.mkString(","))
        logger.debug(s"Distribution of classes for classification after sampling: ${counts.mkString(" , ")} ")

        val (timeBoxCalculationsCancellationFun, timeBoxCalculations) = FutureCancellable {
          logger.info(s"$timeBox launched:")
          currentDataSize = workingDataSet.count()
          logger.info(s"Evolution number $evolutionNumber is launched with datasize = $currentDataSize (rows) out of $totalDataSize (rows) ...")

          def condition = System.currentTimeMillis() - startTime < timeBox.upperBoundary

          var generationNumber = 0
          generationNumberKamon.set(0)

          while (condition) {

            logger.info(s"Time left: $totalTimeLeft")
            logger.info(s"Evolution number still $evolutionNumber - next generation number $generationNumber is launched.")

            import EvaluatedTemplateDataDTOJsonProtocol._
            import spray.json._
            val evolutionProgressDTO = EvolutionProgressDTO(evolutionNumber, generationNumber, timeBox.toString, "startingEvaluation").toJson
            webClientNotifier ! UpdateWebWithJson(evolutionProgressDTO.prettyPrint)

            val evalContext = EvaluationContextInfo(workingDataSet.count(), evolutionNumber, generationNumber)
            templateEvDim.evolveFromLastPopulation(workingDataSet, evalContext)
            logger.info(s"Finished generation number $generationNumber")
            generationNumber += 1
            generationNumberKamon.increment(1)
          }
        } { logger.debug(s"Future { $timeBox } was cancelled !!!!")}

        try {
          Await.result(timeBoxCalculations, restOfTheTimeBox.milliseconds)
        } catch {
          case ex: TimeoutException =>
            timeBoxCalculationsCancellationFun()
            val infoMessage = s"Timeout for $timeBox has happened."
            logger.info(infoMessage + ex.getMessage)
            if (workingDataSet.count() < data.count()) {
              val newDataSize = currentDataSize + evolutionDataSizeIncrement
              logger.info(s"Increasing workingDataSet size for the evolution #${evolutionNumber + 1} to  ${newDataSize}.")

              //TODO Default is new RandomDataSetSizeEvolutionStrategy() it means that on every iteration we will have quite a new distribution of points
              workingDataSet = dataSetSizeEvolutionStrategy.evolve(workingDataSet, newSize = newDataSize, maxEvolutions, data, seed)

            } else {
              logger.info(s"Whole dataset is being used on ${evolutionNumber}th evolution out of maximum $maxEvolutions evolutions")
            }
            // We increasing evolution number whenever we have interrupted current TimeBox
            evolutionNumber += 1
        }
      }
    }

    PopulationHelper.print(templateEvDim.getPopulation, "Final population:")
    
    AutoMLReporter.show(templateEvDim.hallOfFame.clone().dequeueAll.iterator, problemType)
    templateEvDim.hallOfFame
  }

}



package com.automl

import com.automl.dataset._
import com.automl.evolution.dimension.{TemplateEvolutionDimension, TemplateHyperParametersEvolutionDimension}
import com.automl.helper._
import com.automl.problemtype.{ProblemType, ProblemTypeThresholdEstimator}
import com.automl.report.AutoMLReporter
import com.typesafe.scalalogging.LazyLogging
import kamon.Kamon
import org.apache.spark.sql.DataFrame

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.concurrent.{Await, Future, _}

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
             seedPopulation: TPopulation = null,
             initialPopulationSize: Option[Int] = None,
             isBigSizeThreshold: Long = 500,
             isBigDimensionsThreshold: Long = 200,
             initialSampleSize: Long = 500,
             dataSetSizeEvolutionStrategy: DataSetSizeEvolutionStrategy = new RandomDataSetSizeEvolutionStrategy()
            ) extends LazyLogging {

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

  /*  Sampling  */
  def isDataSetBalanced = true // TODO add here concreate estimation of balancing
  implicit val samplingStrategy: SamplingStrategy = if(isDataSetBalanced) new RandomSampling() else new StratifiedSampling()

  val metaDB = new MetaDB() // TODO How it should look like?
  // 100-300 dims,  500 - 5000 examples, num classes,
  // metalerning landmarks(vector of performance of simple fast algorithms-> Set(DT, Bagging{KNN, GBL})
  // Except from statistical metrics we can use base model's performance metrics as a metrics to choose similar datasets.
  // SHould find Euclidian or Manhattan distance between vectors of of this metrics.

  def generateInitialPopulation(size: Int): TPopulation = TPopulation.fromSeedPopulation(seedPopulation)
    .withSize(size)
    .withDefaultMutationProbs
    .build

  def stagnationDetected(evaluationResult: Any): Boolean = {
    // If we run into stagnation?
    // We could check wheter our structures are not changing any more( bad mutation algorithm) or
    // fitness values of our individuals do not improve(or within threshold) when datasize is maximum.
    // We might never reach this state
    false
  } // TODO No stagnation detected for now

  /*Probably we need a tree of dimensions in order to predefine dependencies*/
  def runEvolution(templateEvDim: TemplateEvolutionDimension,
                   hyperParamsEvDim: TemplateHyperParametersEvolutionDimension // TODO unused
                  ): Unit = {

    var workingDataSet: DataFrame = if(isDataBig) {
      samplingStrategy.sample(data, initialSampleSize) //TODO maybe we can start using EvolutionStrategy even here?
    } else data

    var populationOfTemplates: TPopulation = if(useMetaDB) {
      new TPopulation(metaDB.getPopulationOfTemplates)
    } else {
      generateInitialPopulation(initialPopulationSize.get)
    }

    var currentDataSize = workingDataSet.count()
    currentDatasizeKamon.set(currentDataSize)

    val bestEvaluatedTemplatesFromAllGenerationsQueue = collection.mutable.PriorityQueue[EvaluatedTemplateData]()


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

      val timeBoxCalculations = Future {
        logger.info(s"$timeBox launched:")

        def condition = System.currentTimeMillis() - startTime < timeBox.upperBoundary

        while (condition) {

          //In each subsequent evolution, templates are more specific and the percentage of wildcards decrease.
          //For subsequent evolutions, we use population from the last epoch of the previous evolution
          // TODO Also, ranges of explored parameters increase as templates get more precise and specific. ???

          logger.info(s"Evolution number $evolutionNumber is launched with datasize = $currentDataSize (rows) out of $totalDataSize (rows) ...")

          var generationNumber = 0
          generationNumberKamon.set(0)

          var doEscapeFlag = false

          while (condition && generationNumber < maxGenerations && !doEscapeFlag) {

            logger.info(s"Time left: ${(maxTime - System.currentTimeMillis() + startTime) / 1000}")
            logger.info(s"Generation number $generationNumber is launched ( evolution number $evolutionNumber)")

            logger.info("Current population:")
            PopulationHelper.print(populationOfTemplates)

            val evolvedPopulation = templateEvDim.evolve(populationOfTemplates, workingDataSet, problemType)
            // TODO If we stuck here for too long then we are not updating `populationOfTemplates` and starting next generation from scratch.
            populationOfTemplates = evolvedPopulation // TODO maybe we don't need to store it here since we store it in dimension itself

            val bestSurvivedEvaluatedTemplate = templateEvDim.getBestFromPopulation(workingDataSet, problemType)
            //TODO we were putting into queue only best from evolution not from each generation before.
            logger.info(s"Best candidate from  evolution #$evolutionNumber generation #$generationNumber added to priority queue: $bestSurvivedEvaluatedTemplate")
            bestEvaluatedTemplatesFromAllGenerationsQueue.enqueue(bestSurvivedEvaluatedTemplate)

            generationNumber += 1
            generationNumberKamon.increment(1)
          }

          //We increase evolution number only when we change environment(i.e. datasize)
          if (currentDataSize < totalDataSize) {
            // data is doubled (both dimensionality and numerosity if possible).
            // we can increase range of hyperparameters to choose from.
            currentDataSize += evolutionDataSizeFactor
            currentDatasizeKamon.set(currentDataSize)

            workingDataSet = dataSetSizeEvolutionStrategy.evolve(workingDataSet, newSize = currentDataSize, maxEvolutions, data)
            evolutionNumber += 1
            evolutionNumberKamon.increment(1)
            generationNumber = 0
            generationNumberKamon.set(0)
          } else {
            logger.info("We reached maxDataSize and maxNumberOfGenerations")
            doEscapeFlag = true // TODO ?? do we need this
          }
        }
      }

      try {
        Await.result(timeBoxCalculations, restOfTheTimeBox.milliseconds)
      } catch {
        case e: TimeoutException =>
          val infoMessage = s"Timeout for $timeBox has happened."
          logger.info(infoMessage)
          logger.debug(infoMessage + e.getMessage)
      }
    }

    AutoMLReporter.show(bestEvaluatedTemplatesFromAllGenerationsQueue) // TODO we need to return best individual for validation on testSplit
  }

}



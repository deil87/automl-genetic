package com.automl

import com.automl.helper._
import com.automl.template._
import com.automl.template.ensemble.EnsemblingMember
import com.automl.template.simple.SimpleModelMember
import kamon.Kamon
import org.apache.spark.sql.DataFrame

import scala.collection.mutable
import scala.util.Random

class MetaDB() {
  def getPopulationOfTemplates = ???  // Population of base models?
}



class AutoML(data: DataFrame,
             maxTime: Long,
             maxGenerations: Int = 10,
             maxEvolutions: Int = 5,
             useMetaDB: Boolean,
             seedPopulation: Population = Population.firstEverPopulation, // TODO make it optional because of useMetaDB
             initialPopulationSize: Option[Int] = None,
             isBigSizeThreshold: Long = 500,
             isBigDimensionsThreshold: Long = 200,
             initialSampleSize: Long = 500) {

  require(!useMetaDB && initialPopulationSize.isDefined, "If there is no metaDB information then we should start from scratch with population of defined size")

  lazy val totalDataSize: Long = getDataSize(data)

  val evolutionDataSizeFactor: Long = Math.max(totalDataSize / maxEvolutions, 500)

  val timeBoxes: EvolutionTimeBoxes = {
    val strategy: EqualEvolutionsStrategy = EqualEvolutionsStrategy(maxTime, maxEvolutions)
    val builder = EvolutionTimeBoxesBuilder(maxTime, maxEvolutions).withSplittingStrategy(strategy)
    builder.build
  }

  /**
    * Kamon metrics
    */
  import kamon.prometheus.PrometheusReporter
  Kamon.addReporter(new PrometheusReporter())

  val evolutionNumberKamon = Kamon.gauge("kamon.automl.evolution_number")
  val generationNumberKamon = Kamon.gauge("kamon.automl.generation_number")
  val cacheHitsCounterKamon = Kamon.counter("kamon.automl.cache_hits")

  def isDataBig(df: DataFrame): Boolean = {
    def numberOfDimensions: Int = df.columns.length
    numberOfDimensions >= 200 || getDataSize(data) >= isBigSizeThreshold

    //import org.apache.spark.util.SizeEstimator
    //println(SizeEstimator.estimate(distFile))

  }
  def isDataBig(size: Long): Boolean = size >= isBigSizeThreshold

  def getDataSize(df: DataFrame): Long = df.count()

  def sample(df: DataFrame, size: Long): DataFrame = {
    val isBalanced: Boolean = true // TODO add here concreate estimation of balancing
    // TODO We sample both instances and attributes when constrains are violated to get a representative data subset.
    if(isBalanced) {
      //random subsampling
      import org.apache.spark.sql.functions.rand
      df.orderBy(rand()).limit(size.toInt)
    }
    else {
      //stratified subsampling
      ???
    }

  }

  val metaDB = new MetaDB() // TODO How it should look like?
  // 100-300 dims,  500 - 5000 examples, num classes,
  // metalerning landmarks(vector of performance of simple fast algorithms-> Set(DT, Bagging{KNN, GBL})
  // Except from statistical metrics we can use base model's performance metrics as a metrics to choose similar datasets.
  // SHould find Euclidian or Manhattan distance between vectors of of this metrics.

  def generateInitialPopulation(size: Int): Population = Population.fromSeedPopulation(seedPopulation).withSize(size).build

  def stagnationDetected(evaluationResult: Any): Boolean = false // TODO No stagnation detected for now

  def parentSelectionByFitnessRank(selectionShare: Double, individuals: Seq[IndividualAlgorithmData]): Seq[IndividualAlgorithmData] = {
    require(selectionShare < 1 && selectionShare > 0, "Selection share parameter shoud be in range (0, 1)" )
    val numberOfCompetitors = individuals.length
    val numberOfParents = (numberOfCompetitors * selectionShare).toInt
    val orderedByFitness = individuals.sortWith(_.fitness.fitnessError > _.fitness.fitnessError)

    val probabilityStrategy = new LinearRankingProbabilityStrategy(numberOfCompetitors, parameter_S = 1.5)

    val ranked = orderedByFitness
      .zipWithIndex
      .map { case (ind, rank) =>
        ind.withRank(rank).withProbability(probabilityStrategy.computeProbabilityFor(rank))
      }

    val rankedWithCumulativeProbs = ranked.drop(1).scanLeft(ranked.head){ case (acc, indd2) => indd2.copy(probability = indd2.probability + acc.probability)}
    println(rankedWithCumulativeProbs.map(r => (r.rank, r.probability)).mkString("\n"))

    var currentParentIndex = 0
    val selectedParents = new Array[IndividualAlgorithmData](numberOfParents)
    while (currentParentIndex < numberOfParents) {
      val r = Random.nextDouble()
      val rouletteWheel = rankedWithCumulativeProbs.dropWhile(individualData => individualData.probability < r)
      val selected = rouletteWheel.headOption.getOrElse(rankedWithCumulativeProbs.last)
      println(s"Selected for r = $r : $selected")
      selectedParents(currentParentIndex) = selected
      currentParentIndex += 1
    }
    selectedParents // TODO choose optimal data structure. Implicit conversion here.
  }


  def applyMutation(population: Population): Population = {
    print(s"\n\nStarting new mutation phase...")

    def mutate(individual: TemplateTree[TemplateMember]) = {

      val individualsTreeHeight = individual.height
      val initialProbability, probStep: Double = 1.0 / individualsTreeHeight
      println(s"Initial probability: $initialProbability, probStep: $probStep")
      val mutationProbabilities = MutationProbabilities(initialProbability)

      def getRandomEnsemblingMember = EnsemblingMember.poolOfEnsemblingModels.toSeq.randElement
      def getRandomBaseMember: TemplateMember = SimpleModelMember.poolOfSimpleModels.randElement

      val structOrMemberThreshold = 0.5
      //val structThreshold = 0.5

      def traverseAndMutate(individual: TemplateTree[TemplateMember], mutProbs: MutationProbabilities): TemplateTree[TemplateMember] = individual match {
        case lt@LeafTemplate(_) =>
          if (structOrMemberThreshold >= Random.nextDouble()) {
            if (mutProbs.structureProb >= Random.nextDouble()) {

              val numberOfNewBaseModels = Random.nextInt(4) + 1 // TODO parameter?

              val newMember = NodeTemplate(getRandomEnsemblingMember, Seq(lt) ++ (0 to numberOfNewBaseModels).map(_ => LeafTemplate(getRandomBaseMember)))
              println(s"\nStructural mutation happened for $lt --> $newMember")
              newMember
            }
            else if (mutProbs.memberProb >= Random.nextDouble()) {
              val newMember: TemplateMember = getRandomBaseMember
              println(s"\nMember mutation happened for $lt --> $newMember")
              LeafTemplate(newMember)
            }
            else lt
          }
          else lt

        case nt@NodeTemplate(ensemblingMember, subMembers) =>
          val updatedMutationProbs = mutProbs.increaseAllBy(probStep)

          NodeTemplate(ensemblingMember, subMembers.map(traverseAndMutate(_, updatedMutationProbs)))

      }

      traverseAndMutate(individual, mutationProbabilities)
    }

    new Population(population.individuals map mutate)

  }

  val individualsCache = mutable.Map[(TemplateTree[TemplateMember], Long), FitnessResult]()

  def calculateFitnessResults(population: Population, workingDataSet: DataFrame): Seq[IndividualAlgorithmData] = {
    population.individuals.zipWithIndex.map{ case (individualTemplate, idx) => (idx, individualTemplate, TemplateTreeHelper.materialize(individualTemplate))}
      .map{ case (idx, template, materializedTemplate) =>

        val cacheKey = (materializedTemplate, workingDataSet.count())
        if (individualsCache.isDefinedAt(cacheKey)) {
          println(s"Cache hit happened for $idx individual based on: \n template: $template \n algorithm: $materializedTemplate \n")
          cacheHitsCounterKamon.increment(1)
        }


        val fitness = individualsCache.getOrElseUpdate(cacheKey, {
          println(s"Calculated new value for $idx individual based on: \n template: $template \n algorithm: $materializedTemplate \n")
          // TODO can we split it randomly here???
          val Array(trainingSplit, testSplit) = workingDataSet.randomSplit(Array(0.67, 0.33), 11L)
          materializedTemplate.evaluateFitness(trainingSplit, testSplit)
        }) // TODO unbounded addition. Memory leak
        val iad = IndividualAlgorithmData(idx.toString, template, materializedTemplate, fitness)
        iad.sendMetric()
        iad
      }
  }

  def chooseBestTemplate(population: Population, workingDataSet: DataFrame): IndividualAlgorithmData = {
    val individualsWithEvaluationResults = calculateFitnessResults(population, workingDataSet)

    individualsWithEvaluationResults.sortWith(_.fitness.fitnessError < _.fitness.fitnessError).head
  }

  def run(): Unit = {

    var workingDataSet: DataFrame = if(isDataBig(data)) {
      sample(data, initialSampleSize)
    } else data

    var individualsTemplates = if(useMetaDB) {
      //With the metadatabase, the initial population is filled by best individuals from most similar meta data
      // (pair- wise similarities of attributes statistics)

      // The probability that a template is selected for seeding the population is inversely proportional to the
      // squared distance of meta data vectors and proportional to a robust performance of the template.
      //The robust performance is defined as average rank of template performance on similar data sets.
      // Then, as the algorithm runs, templates consisting of one base algorithm are evaluated on the data set
      // and stored into the metadatabase. Their performance is used as landmarking attribute [22] and together with
      // data statistics make up meta-features. The meta-features vector is then compared to other vectors stored in
      // the metadatabase and the most similar records are returned.
      new Population(metaDB.getPopulationOfTemplates)
    } else {
      //In case that the metadatabase is not used, base models form the population.
      //The advantage is that each type of base model is considered before ensembles are taken into account
      generateInitialPopulation(initialPopulationSize.get)
    }

    var currentDataSize = initialSampleSize

    val bestIndividualsFromAllEvolutions = collection.mutable.PriorityQueue[IndividualAlgorithmData]()


    //The fitness of each template is updated during evolutions and when the optimization terminates,
    // winning templates are saved as a new record into the metadatabase or corresponding records are
    // updated with the new templates.
    var evolutionNumber = 0
    evolutionNumberKamon.set(0)

    timeBoxes.timeBoxes foreach { timeBox =>
      println(s"TimeBox # ${timeBox.index} launched:")

      val startTime = System.currentTimeMillis()

      //While time is available, we run a sequence of evolutions that are gradually
      //exploring the state space of possible templates.
      def condition = System.currentTimeMillis() - startTime < timeBox.duration

      while (condition) {

        //In each subsequent evolution, templates are more specific and the percentage of wildcards decrease.
        //For subsequent evolutions, we use population from the last epoch of the previous evolution
        // TODO Also, ranges of explored parameters increase as templates get more precise and specific. ???

        println(s"LAUNCHING evolutionNumber=$evolutionNumber with datasize= $currentDataSize out of $totalDataSize ...")

        var generationNumber = 0
        generationNumberKamon.set(0)

        var doEscapeFlag = false

        while (condition && generationNumber < maxGenerations && !doEscapeFlag) {
          //TODO it is a good idea to cache results for similar templates/models

          println(s"Time left: ${(maxTime - System.currentTimeMillis() + startTime) / 1000}")
          println(s"LAUNCHING evolutionNumber=$evolutionNumber generationNumber=$generationNumber...")
          PopulationHelper.print(individualsTemplates)


          // c) fitness function formulation
          //It is estimated by a multiple crossvalidation (CV) [83,81]
          //The fitness of a template is proportional to the average performance of models generated on training folds
          // and evaluated on testing folds, while the data is divided into folds multiple times.

          val evaluatedIndividuals: Seq[IndividualAlgorithmData] =
            calculateFitnessResults(individualsTemplates, workingDataSet)

          evaluatedIndividuals.zipWithIndex.sortBy(_._1.fitness.fitnessError).map { case (indivData, idx) =>
            (idx, s"$idx) ${indivData.fitness.fitnessError} \n")
          }.sortBy(_._1).foreach { case (_, str) => println(str) }


          // GL(t) = 100 * (  Eva(t)/ Eopt(t) - 1 )
          //        High generalization loss is one obvious candidate reason to stop training, because
          //        it directly indicates overfitting. This leads us to the 1rst class of stopping
          //        criteria: stop as soon as the generalization loss exceeds a certain
          //          threshold. We define the class GL as
          //        GL : stop after first epoch t with GL(t) > alpha
          //How can we do this multiple time? In parallel?

          if (evolutionNumber == 0 && !useMetaDB) {
            //putInfoInto MetaDB
          }


          // If we run into stagnation?
          // We could check wheter our structures are not changing any more( bad mutation algorithm) or
          // fitness values of our individuals do not improve(or within threshold) when datasize is maximum.
          // We might never reach this state
          if (stagnationDetected(evaluatedIndividuals)) // we are breaking our while loop - early stopping?
            generationNumber = maxGenerations

          //Second phase: We are going to compute fitness functions and rank all the individuals.
          // Draw from these population with the probability distribution proportional to rank values.
          val individualsForMutation = parentSelectionByFitnessRank(0.5, evaluatedIndividuals).map(_.template)

          //b) design of genetic operators and evolution
          //Parameters of a node are mutated by applying Gaussian noise to the current value
          //We do not use crossover, just mutations similar to the approach used in a standard GP

          // First phase: recombination or mutation. Ensure that diversity is ok .
          // Maybe check child/mutant on ability to survive by itself. Then ensure that the size of population is ok
          // What kind of parameters we mutate
          val offspring = applyMutation(new Population(individualsForMutation)) // individualsForMutation

          val subjectsToSurvival = new Population(individualsTemplates.individuals ++ offspring.individuals)
          val evaluationResultsForAll = calculateFitnessResults(subjectsToSurvival, workingDataSet)

          //Select 50% best of all the (individuals + offspring)
          val survivedForNextGenerationSeed: Seq[TemplateTree[TemplateMember]] = parentSelectionByFitnessRank(0.5, evaluationResultsForAll).map(_.template)
          individualsTemplates = Population.fromSeedPopulation(new Population(survivedForNextGenerationSeed)).withSize(initialPopulationSize.get).build
          generationNumber += 1
          generationNumberKamon.increment(1)
        }


        //How can we call next size level - evolution?
        if (currentDataSize < totalDataSize) {
          // data is doubled (both dimensionality and numerosity if possible).
          // we can increase range of hyperparameters to choose from.
          currentDataSize += evolutionDataSizeFactor
          workingDataSet = sample(data, if (currentDataSize >= totalDataSize) totalDataSize else currentDataSize) // TODO should we sample new or append to previous data some new sample?
          evolutionNumber += 1
          evolutionNumberKamon.increment(1)
          generationNumber = 0
          generationNumberKamon.set(0)
        } else {
          // We've got individuals survived after being traind on whole datasets.
          // Next step is to refine results and choose the best argorithm that we run into on our way here.
          // We should skip 'datasize loop' and 'time loop' and call chooseBestTemplate
          println("We reached maxDataSize and maxNumberOfGenerations")
          doEscapeFlag = true
          //increase validation part of data. Start using k-folds CV. Of-by-one validation. then multiple round of CV.
          // in general we start using as much data as possible to prevent overfitting and decrease generalization error.
        }

        val bestFromCurrentEvolution = chooseBestTemplate(individualsTemplates, workingDataSet)
        println(s"Best candidate from  evolution #${evolutionNumber - 1} added to priority queue: $bestFromCurrentEvolution")
        bestIndividualsFromAllEvolutions.enqueue(bestFromCurrentEvolution)
      }

    }
    // Final evaluation on different test data. Consider winners from all evolutions(evolutionNumbers) but put more faith into last ones because they have been chosen based on results on bigger  validation sets(better representative of a population).
    val winner = bestIndividualsFromAllEvolutions.dequeue()

    println("\n##############################################################")
    println("Fitness value of the BEST template: " +  winner.fitness.fitnessError)
    println("Best template: " + TemplateTreeHelper.print2(winner.template)) // TODO make print2 actually a printing method
    println("Other best individuals results:\n" + bestIndividualsFromAllEvolutions.dequeueAll.map(_.fitness.fitnessError).mkString(",\n"))

  }


  // TODO Testing with UCI datasets?

  // How should look our metaDB store? is it mapping of multiple criteria to a template?
  // How should look initial form?



}



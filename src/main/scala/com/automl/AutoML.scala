package com.automl

import com.automl.helper._
import com.automl.template._
import com.automl.template.ensemble.EnsemblingMember
import com.automl.template.simple.SimpleModelMember
import org.apache.spark.sql.DataFrame

import scala.collection.mutable
import scala.util.Random

class MetaDB() {
  def getPopulationTemplate = ???  // Population of base models?
}



class AutoML(data: DataFrame,
             maxTime: Long,
             maxGenerations: Int = 10,
             useMetaDB: Boolean,
             seedPopulation: Population = Population.firstEverPopulation, // TODO make it optional because of useMetaDB
             initialPopulationSize: Option[Int] = None,
             isBigSizeThreshold: Long = 500,
             isBigDimensionsThreshold: Long = 200,
             initialSampleSize: Long = 500) {

  require(!useMetaDB && initialPopulationSize.isDefined, "If there is no metaDB information then we should start from scratch with population of defined size")


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

  def parentSelectionByFitnessRank(selectionShare: Double, individuals: Seq[(TemplateTree[TemplateMember], TemplateTree[TemplateMember], FitnessResult)]): Seq[IndividualData] = {
    require(selectionShare < 1 && selectionShare > 0, "Selection share parameter shoud be in range (0, 1)" )
    val numberOfCompetitors = individuals.length
    val numberOfParents = (numberOfCompetitors * selectionShare).toInt
    val orderedByFitness = individuals.sortWith(_._3.fitnessError > _._3.fitnessError)

    val probabilityStrategy = new LinearRankingProbabilityStrategy(numberOfCompetitors, parameter_S = 1.5)

    val ranked = orderedByFitness.zipWithIndex
      .map{ case ((ind, alg, fr), rank) => IndividualData(ind, alg, fr, rank, probabilityStrategy.computeProbabilityFor(rank))}
    val rankedWithCumulativeProbs = ranked.drop(1).scanLeft(ranked.head){ case (acc, indd2) => indd2.copy(probability = indd2.probability + acc.probability)}
    println(rankedWithCumulativeProbs.map(r => (r.rank, r.probability)).mkString("\n"))

    var currentParentIndex = 0
    val selectedParents = new Array[IndividualData](numberOfParents)
    while (currentParentIndex < numberOfParents) {
      val r = Random.nextDouble()
      val rouletteWheel = rankedWithCumulativeProbs.dropWhile(individualData => individualData.probability < r)
      val selected = rouletteWheel.headOption.getOrElse(rankedWithCumulativeProbs.last)
      println(s"Selected for r = $r : ${selected}")
      selectedParents(currentParentIndex) = selected
      currentParentIndex += 1
    }
    selectedParents // TODO choose optimal data structure. Implicit conversion here.
  }


  def applyMutation(population: Population): Population = {
    print(s"\n\nStarting new mutation phase...")

    def mutate(individual: TemplateTree[TemplateMember]) = {
      var structureMutationUnits = 1
      var memberMutationUnits = 1

      val individualsTreeHeight = individual.height
      val initialProbability, probStep: Double = 1 / individualsTreeHeight
      var probabilityOfStructMutation, probabilityOfMemberMutation = Probability(initialProbability)

      def getRandomEnsemblingMember = EnsemblingMember.poolOfEnsemblingModels.toSeq.randElement
      def getRandomBaseMember: TemplateMember = (SimpleModelMember.poolOfSimpleModels).randElement
      //TODO do we need this?
      def getRandomAnyMember: TemplateMember = (EnsemblingMember.poolOfEnsemblingModels.toSeq ++ SimpleModelMember.poolOfSimpleModels).randElement

      val structOrMemberThreshold = 0.5

      def traverseAndMutate(individual: TemplateTree[TemplateMember]): TemplateTree[TemplateMember] = individual match {
        case lt@LeafTemplate(_) =>
          if (structOrMemberThreshold >= Random.nextDouble() && structureMutationUnits > 0 && probabilityOfStructMutation >= Random.nextDouble()) {
            structureMutationUnits -= 1
            val numberOfNewBaseModels = Random.nextInt(4) + 1
            NodeTemplate(getRandomEnsemblingMember, Seq(lt) ++ (0 to numberOfNewBaseModels).map(_ => LeafTemplate(getRandomBaseMember)))
          }
          else if (memberMutationUnits > 0 && probabilityOfMemberMutation >= Random.nextDouble()) {
            memberMutationUnits -= 1
            val newMember: TemplateMember = getRandomBaseMember
            println(s"\nMember mutation happened for $lt --> $newMember")
            LeafTemplate(newMember)
          }
          else {
            lt
          }
        case nt@NodeTemplate(ensemblingMember, subMembers) =>
          if (structureMutationUnits > 0 || memberMutationUnits > 0) {
            probabilityOfStructMutation = probabilityOfStructMutation.increaseBy(probStep)
            probabilityOfMemberMutation = probabilityOfMemberMutation.increaseBy(probStep)

            NodeTemplate(ensemblingMember, subMembers.map(traverseAndMutate))
          }
          else {
            nt
          }
      }

      traverseAndMutate(individual)
    }

    new Population(population.individuals map mutate)

  }

  val individualsCache = mutable.Map[TemplateTree[TemplateMember], FitnessResult]()

  def calculateFitnessResults(individuals: Seq[(TemplateTree[TemplateMember])], workingDataSet: DataFrame): Seq[(TemplateTree[TemplateMember], TemplateTree[TemplateMember], FitnessResult)] = {
    individuals.map(individualTemplate => (individualTemplate, TemplateTreeHelper.materialize(individualTemplate)))
      .map{ case (template, materializedTemplate) =>
        val Array(trainingSplit,testSplit)  = data.randomSplit(Array(0.67, 0.33),11L)
        val fitness = individualsCache.getOrElseUpdate(materializedTemplate, materializedTemplate.evaluateFitness(trainingSplit, testSplit)) // TODO unbounded addition. Memory leak
        (template, materializedTemplate, fitness)
      }
  }

  def chooseBestTemplate(individuals: Seq[(TemplateTree[TemplateMember])], workingDataSet: DataFrame): (TemplateTree[TemplateMember], TemplateTree[TemplateMember], FitnessResult) = {
    val individualsWithEvaluationResults = calculateFitnessResults(individuals, workingDataSet)
    individualsWithEvaluationResults.sortWith(_._3.fitnessError < _._3.fitnessError).head
  }

  def run(): Unit = {
    val startTime = System.currentTimeMillis()

    val dataSize = getDataSize(data)

    var workingDataSet: DataFrame = if(isDataBig(data)) {
      sample(data, initialSampleSize)
    } else data

    var individuals = if(useMetaDB) {
      //With the metadatabase, the initial population is filled by best individuals from most similar meta data
      // (pair- wise similarities of attributes statistics)

      // The probability that a template is selected for seeding the population is inversely proportional to the
      // squared distance of meta data vectors and proportional to a robust performance of the template.
      //The robust performance is defined as average rank of template performance on similar data sets.
      // Then, as the algorithm runs, templates consisting of one base algorithm are evaluated on the data set
      // and stored into the metadatabase. Their performance is used as landmarking attribute [22] and together with
      // data statistics make up meta-features. The meta-features vector is then compared to other vectors stored in
      // the metadatabase and the most similar records are returned.
      metaDB.getPopulationTemplate
    } else {
      //In case that the metadatabase is not used, base models form the population.
      //The advantage is that each type of base model is considered before ensembles are taken into account
      generateInitialPopulation(initialPopulationSize.get).individuals
    }

    var currentDataSize = initialSampleSize

    //The fitness of each template is updated during evolutions and when the optimization terminates,
    // winning templates are saved as a new record into the metadatabase or corresponding records are
    // updated with the new templates.
    var evolutionNumber = 0

    implicit val individualsOrdering = new Ordering[(TemplateTree[TemplateMember], TemplateTree[TemplateMember], FitnessResult)] {
      override def compare(x: (TemplateTree[TemplateMember], TemplateTree[TemplateMember], FitnessResult), y: (TemplateTree[TemplateMember], TemplateTree[TemplateMember], FitnessResult)) = {
        if(x._3.fitnessError < y._3.fitnessError) 1
        else if(x._3.fitnessError > y._3.fitnessError) -1
        else 0
      }
    }

    val bestIndividualsFromAllEvolutions = collection.mutable.PriorityQueue[(TemplateTree[TemplateMember], TemplateTree[TemplateMember], FitnessResult)]()


    //While time is available, we run a sequence of evolutions that are gradually
    //exploring the state space of possible templates.
    def condition = System.currentTimeMillis() - startTime < maxTime
    while (condition) {

      //In each subsequent evolution, templates are more specific and the percentage of wildcards decrease.
      //For subsequent evolutions, we use population from the last epoch of the previous evolution
      // TODO Also, ranges of explored parameters increase as templates get more precise and specific. ???

      println(s"LAUNCHING evolutionNumber=${evolutionNumber} with datasize= ${currentDataSize} out of ${dataSize} ...")

      var generationNumber = 0

      var doEscapeFlag = false

      while (condition && generationNumber < maxGenerations && !doEscapeFlag) {
        //TODO it is a good idea to cache results for similar templates/models

        println(s"Time left: ${(maxTime - System.currentTimeMillis() + startTime) / 1000}")
        println(s"LAUNCHING generationNumber=${generationNumber}...")
        PopulationHelper.print(new Population(individuals))


        // c) fitness function formulation
        //It is estimated by a multiple crossvalidation (CV) [83,81]
        //The fitness of a template is proportional to the average performance of models generated on training folds
        // and evaluated on testing folds, while the data is divided into folds multiple times.

        val evaluatedIndividuals: Seq[(TemplateTree[TemplateMember], TemplateTree[TemplateMember], FitnessResult)] =
          calculateFitnessResults(individuals, workingDataSet)

        evaluatedIndividuals.zipWithIndex.sortBy(_._1._3.fitnessError).map{ case ((inds, materialised, fitness), idx) =>
          (idx, s"${idx}) ${fitness.fitnessError} \n")
        }.sortBy(_._1).foreach{ case (_, str) => println(str)}


        // GL(t) = 100 * (  Eva(t)/ Eopt(t) - 1 )
//        High generalization loss is one obvious candidate reason to stop training, because
//        it directly indicates overfitting. This leads us to the 1rst class of stopping
//        criteria: stop as soon as the generalization loss exceeds a certain
//          threshold. We define the class GL as
//        GL : stop after first epoch t with GL(t) > alpha
        //How can we do this multiple time? In parallel?

        if(evolutionNumber == 0 && !useMetaDB) {
          //putInfoInto MetaDB
        }


        // Is stagnation a convergence?
        // Stagnation for which individual? For all in population?
        // We might never reach this state
        if( stagnationDetected(evaluatedIndividuals)) // we are breaking our while loop - early stopping?
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

        val subjectsToSurvival = individuals ++ offspring.individuals
        val evaluationResultsForAll = calculateFitnessResults(subjectsToSurvival, workingDataSet)

        //Select 50% best of all the (individuals + offspring)
        val survivedForNextGenerationSeed: Seq[TemplateTree[TemplateMember]] = parentSelectionByFitnessRank(0.5, evaluationResultsForAll).map(_.template)
        individuals = Population.fromSeedPopulation(new Population(survivedForNextGenerationSeed)).withSize(initialPopulationSize.get).build.individuals
        generationNumber += 1
      }


      //How can we call next size level - evolution?
      if(currentDataSize < dataSize) {
        // data is doubled (both dimensionality and numerosity if possible).
        // we can increase range of hyperparameters to choose from.
        currentDataSize *=2
        workingDataSet = sample(data, currentDataSize) // TODO should we sample new or append to previous data some new sample?
        evolutionNumber += 1
        generationNumber = 0
      } else if(currentDataSize > dataSize) {
        currentDataSize = dataSize
      } else {
        // We've got individuals survived after being traind on whole datasets.
        // Next step is to refine results and choose the best argorithm that we run into on our way here.
        // We should skip 'datasize loop' and 'time loop' and call chooseBestTemplate
        doEscapeFlag = true
        //increase validation part of data. Start using k-folds CV. Of-by-one validation. then multiple round of CV.
        // in general we start using as much data as possible to prevent overfitting and decrease generalization error.
      }

      val bestFromCurrentEvolution = chooseBestTemplate(individuals, workingDataSet)
      println(s"Best candidate from  evolution #${evolutionNumber - 1} added to priority queue: $bestFromCurrentEvolution")
      bestIndividualsFromAllEvolutions.enqueue(bestFromCurrentEvolution)
    }

    // Final evaluation on different test data. Consider winners from all evolutions(evolutionNumbers) but put more faith into last ones because they have been chosen based on results on bigger  validation sets(better representative of a population).
    val winner = bestIndividualsFromAllEvolutions.dequeue()

    println("\n##############################################################")
    println("Fitness value of the BEST template: " +  winner._3.fitnessError)
    println("Best template: " + TemplateTreeHelper.print2(winner._1)) // TODO make print2 actually a printing method
    println("Other best individuals results:\n" + bestIndividualsFromAllEvolutions.dequeueAll.map(_._3.fitnessError).mkString(",\n"))

  }


  // TODO Testing with UCI datasets?

  // How should look our metaDB store? is it mapping of multiple criteria to a template?
  // How should look initial form?



}



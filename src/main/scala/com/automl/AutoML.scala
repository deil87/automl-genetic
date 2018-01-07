package com.automl

import com.automl.algorithm._
import com.automl.helper.{FitnessResult, Probability, TemplateTreeHelper}
import com.automl.template._
import com.automl.template.ensemble.EnsemblingMember
import com.automl.template.simple.SimpleModelMember
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{PipelineStage, Predictor}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.DataFrame

import scala.util.Random

class MetaDB() {
  def getPopulationTemplate = ???  // Population of base models?
}



class AutoML(data: DataFrame,
             maxTime: Long,
             maxGenerations: Int = 100,
             useMetaDB: Boolean,
             isBigSizeThreshold: Long = 500,
             isBigDimensionsThreshold: Long = 200,
             initialSampleSize: Long = 500) {


  def isDataIsBig(df: DataFrame): Boolean = {
    def numberOfDimensions: Int = ???
    numberOfDimensions >= 200 || getDataSize(data) >= isBigSizeThreshold

  }
  def isDataIsBig(size: Long): Boolean = size >= isBigSizeThreshold

  def getDataSize(df: DataFrame): Long = df.count()

  def sample(df: DataFrame, size: Long): DataFrame = {
    val isBalanced: Boolean = ???
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

  def generateBasePopulation: Seq[TemplateTree[TemplateMember]] = Population.firstEverPopulation

  def evaluateFitnessOfIndividualsOnDataSample(individuals: Set[AlgorithmTree[AlgorithmMember]], sample: DataFrame) = ???

  def stagnationDetected(evaluationResult: Any): Boolean = ???

  def selectIndividualsInTournament(selectionShare: Double, individuals: Seq[(TemplateTree[TemplateMember], AlgorithmTree[AlgorithmMember], FitnessResult)]): Seq[(TemplateTree[TemplateMember], AlgorithmTree[AlgorithmMember], FitnessResult)] = {
    require(selectionShare < 1 && selectionShare > 0, "Selection share parameter shoud be in range (0, 1)" )
    val numberOfCompetitors = individuals.length
    val numberOfWinners = (numberOfCompetitors * selectionShare).toInt
    individuals.sortWith(_._3.fitnessError > _._3.fitnessError).take(numberOfWinners)
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
      def getRandomBaseMember: TemplateMember = (EnsemblingMember.poolOfEnsemblingModels.toSeq ++ SimpleModelMember.poolOfSimpleModels).randElement

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
            println(s"\nMember mutation hapened for $lt --> $newMember")
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

  def prepareNewGeneration(individuals: Set[Any]): Set[Any] = ???

  def chooseBestTemplate = ???

  def run(): Unit = {
    val startTime = System.currentTimeMillis()

    val dataSize = getDataSize(data)

    var workingDataSet: DataFrame = if(isDataIsBig(data)) {
      sample(data, initialSampleSize)
    } else data

    val initialPopulation = if(useMetaDB) {
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
      generateBasePopulation
    }

    var currentDataSize = initialSampleSize

    //While time is available, we run a sequence of evolutions that are gradually
    //exploring the state space of possible templates.
    while (System.currentTimeMillis() - startTime > maxTime) {

      //In each subsequent evolution, templates are more specific and the percentage of wildcards decrease.
      //For subsequent evolutions, we use population from the last epoch of the previous evolution
      // TODO Also, ranges of explored parameters increase as templates get more precise and specific. ???

      //The fitness of each template is updated during evolutions and when the optimization terminates,
      // winning templates are saved as a new record into the metadatabase or corresponding records are
      // updated with the new templates.
      var evolutionNumber = 0

      var generationNumber = 0

      //a) representation of individual
      var individuals: Seq[TemplateTree[TemplateMember]] = initialPopulation

      while (generationNumber < maxGenerations) {
        //it is a good idea to cache results for similar templates/models


        // c) fitness function formulation
        //It is estimated by a multiple crossvalidation (CV) [83,81]
        //The fitness of a template is proportional to the average performance of models generated on training folds
        // and evaluated on testing folds, while the data is divided into folds multiple times.

        val evaluationResults: Seq[(TemplateTree[TemplateMember], AlgorithmTree[AlgorithmMember], FitnessResult)] = individuals
          .map(individualTemplate => (individualTemplate, TemplateTreeHelper.materialize(individualTemplate)))
          .map{ case (template, algorithm) => (template, algorithm, algorithm.evaluate(workingDataSet))}

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
        if( stagnationDetected(evaluationResults)) // we are breaking our while loop - early stopping?
          generationNumber = maxGenerations

        // Second phase: selection. Survival of the fittest ( in terms of evaluated values by fitness function)
        // How can we do this? Just get rid of individuals with maximum validation error? What percent of total will survive?
        val survivals = selectIndividualsInTournament(0.5, evaluationResults).map(_._1)

        //b) design of genetic operators and evolution
        //Parameters of a node are mutated by applying Gaussian noise to the current value
        //We do not use crossover, just mutations similar to the approach used in a standard GP

        // First phase: recombination or mutation. Ensure that diversity is ok .
        // Maybe check child/mutant on ability to survive by itself. Then ensure that the size of population is ok
        // What kind of parameters we mutate
        val mutated = applyMutation(new Population(survivals))

        // What we do to prepare new generation?
        val newIndividualsOriginatedFromSurvivors = prepareNewGeneration(???)  // d) construction of initial population
        individuals = ??? // newIndividualsOriginatedFromSurvivors
        generationNumber += 1
      }


      //How can we call next size level - evolution?
      if(currentDataSize < dataSize) {
        // data is doubled (both dimensionality and numerosity if possible).
        // we can increase range of hyperparameters to choose from.
        currentDataSize *=2
        workingDataSet = sample(data, currentDataSize)
        evolutionNumber += 1
      } else {
        //increase validation part of data. Start using k-folds CV. Of-by-one validation. then multiple round of CV.
        // in general we start using as much data as possible to prevent overfitting and decrease generalization error.
      }



    }

    // Final evaluation on different test data. Consider winners from all evolutions but put more faith into last ones because they have been chosen based on results oo bigger  validation sets(more representative of a population).
    chooseBestTemplate

  }


  // TODO Testing with UCI datasets?

  // How should look our metaDB store? is it mapping of multiple criteria to a template?
  // How should look initial form?



}



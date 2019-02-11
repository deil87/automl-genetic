package com.automl.evolution.dimension
import akka.actor.{ActorRef, ActorSystem}
import com.automl.EvaluatedTemplateData.logger
import com.automl.evolution.dimension.hparameter.{EvaluatedHyperParametersField, HyperParametersField, TemplateHyperParametersEvolutionDimension}
import com.automl.evolution.diversity.{DistinctDiversityStrategy, MisclassificationDistance}
import com.automl.evolution.evaluation.{TemplateNSLCEvaluator, TemplateSimpleEvaluator}
import com.automl.{EvaluatedTemplateData, Population, TPopulation}
import com.automl.evolution.mutation.{DepthDependentTemplateMutationStrategy, MutationProbabilities}
import com.automl.evolution.selection.RankSelectionStrategy
import com.automl.helper.{FitnessResult, PopulationHelper}
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.{TemplateMember, TemplateTree}
import com.typesafe.config.ConfigFactory
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.param.Params
import org.apache.spark.sql.DataFrame

import scala.collection.mutable

/**
  *
  * @param evolveEveryGenerations is not used for now
  * @param problemType We need to take into account which models we can mutate into filtered out by problemType
  */
class TemplateEvolutionDimension(initialPopulation: Option[TPopulation] = None, evolveEveryGenerations: Int = 1, problemType: ProblemType)(implicit val as: ActorSystem)
    extends EvolutionDimension[TPopulation, TemplateTree[TemplateMember], EvaluatedTemplateData]
    with LazyLogging{

  val tdConfig = ConfigFactory.load().getConfig("evolution.templateDimension")

  lazy val evolutionDimensionLabel: String = tdConfig.getString("name")
  lazy val populationSize: Int = tdConfig.getInt("populationSize")

  val distinctStrategy = new DistinctDiversityStrategy()
  val mutationStrategy = new DepthDependentTemplateMutationStrategy(distinctStrategy, problemType)
  val rankSelectionStrategy = new RankSelectionStrategy

  // Dependencies on other dimensions. Hardcoded for now. Should come from AutoML.runEvolution method parameters.
  val hyperParamsEvDim = new TemplateHyperParametersEvolutionDimension(this,problemType = problemType)

  override var _population: TPopulation = new TPopulation(Nil)

  val evaluator = if(problemType == MultiClassClassificationProblem) {
     new TemplateNSLCEvaluator(new MisclassificationDistance)
  } else ???

   // TODO make it faster with reference to value
  override implicit val individualsEvaluationCache = mutable.Map[(TemplateTree[TemplateMember], Long), FitnessResult]()

  implicit val individualsEvaluationCacheExtended = mutable.Map[(TemplateTree[TemplateMember], HyperParametersField, Long), FitnessResult]()

  override val hallOfFame: mutable.PriorityQueue[EvaluatedTemplateData] = collection.mutable.PriorityQueue[EvaluatedTemplateData]()


  override def getInitialPopulationFromMetaDB: TPopulation = ??? //new TPopulation(metaDB.getPopulationOfTemplates)

  override def getInitialColdStartPopulation: TPopulation = {
    initialPopulation.map{population =>
      TPopulation.fromSeedPopulation(population)
        .withSize(populationSize)
        .withDefaultMutationProbs
        .build
    }.getOrElse(throw new IllegalStateException("Initial population was not specified")) // TODO we should provide default populatin here
  }

  //Almost generalisable. Need to specify type that is common to _.template and _.field
  override def getBestFromHallOfFame: TemplateTree[TemplateMember] = hallOfFame.headOption.map(_.template).getOrElse{getInitialPopulation.individuals.head}

  override def updateHallOfFame(evaluatedIndividuals: Seq[EvaluatedTemplateData]): Unit = ???

  override def showCurrentPopulation(): Unit = {
    if(getEvaluatedPopulation.nonEmpty)
      logger.debug(PopulationHelper.renderEvaluatedIndividuals(getEvaluatedPopulation))
    else PopulationHelper.print(getPopulation, "Current population without evaluations")
  }

  var skipEvolutionCountDown: Int = evolveEveryGenerations - 1

  override def evolve(population: TPopulation, workingDF: DataFrame): TPopulation = {

    //TODO Control should be outside of Dimension concept
    //Maybe it is better to set up flag for corner case like 'always execute'
    if(skipEvolutionCountDown > 0) {
      logger.debug(s"SKIPPING evolution. Next evolution in $skipEvolutionCountDown attempts")
      skipEvolutionCountDown -= 1
      return getPopulation
    }
    skipEvolutionCountDown = evolveEveryGenerations - 1

    showCurrentPopulation()

    val evaluatedOriginalPopulation = { //TODO generalize method
      if(getEvaluatedPopulation.nonEmpty) {
        logger.debug("Taking evaluated population from previous generation.")
        getEvaluatedPopulation
      } else {
        logger.debug("Evaluating population for the very first time.")
        evaluatePopulation(population, workingDF)
      }
    }

    //Need to decide where selecting neighbours should go. To evaluation or selection or to its own phase.
    val evaluatedOriginalPopulationWithNeighbours = evaluator.findNeighbours(evaluatedOriginalPopulation, evaluatedOriginalPopulation, population.size)

    val selectedParents = selectParents(evaluatedOriginalPopulationWithNeighbours)

    //Problem: initial ot on the way duplication of individuals. Are we allowed repetitions at all? For now lets keep try to force diversity on mutation phase.
    val populationForUpcomingMutation = extractIndividualsFromEvaluatedIndividuals(selectedParents)

    val offspring = mutateParentPopulation(populationForUpcomingMutation)

    val mutantsEvaluationsForOffspring = evaluatePopulation(offspring, workingDF)

    val evaluatedOffspringWithNeighbours = evaluator.findNeighbours(mutantsEvaluationsForOffspring, mutantsEvaluationsForOffspring ++ evaluatedOriginalPopulationWithNeighbours, population.size)

    val evaluationResultsForNewExpandedGeneration = evaluatedOffspringWithNeighbours ++ evaluatedOriginalPopulationWithNeighbours

    val survivedForNextGenerationEvaluatedTemplates = selectSurvived(population.size, evaluationResultsForNewExpandedGeneration)

    _evaluatedPopulation = survivedForNextGenerationEvaluatedTemplates

    val evolvedPopulation = new TPopulation(survivedForNextGenerationEvaluatedTemplates.map(_.template), offspring.mutationProbabilities)

    // Do backpropagation of fitness. Evolve other dimensions by using new evaluations/best templates
    hyperParamsEvDim.evolveFromLastPopulation(workingDF)

    _population = evolvedPopulation
    evolvedPopulation
  }

  override def selectSurvived(populationSize: Int, evaluationResultsForNewExpandedGeneration: Seq[EvaluatedTemplateData]) = {
    rankSelectionStrategy.parentSelectionBySizeWithLocalCompetitions(populationSize, evaluationResultsForNewExpandedGeneration)
  }


  override def extractIndividualsFromEvaluatedIndividuals(evaluatedIndividuals: Seq[EvaluatedTemplateData]): TPopulation = {
    new TPopulation(evaluatedIndividuals.map(_.template), null) // population.mutationProbabilities
  }

  override def selectParents(evaluated: Seq[EvaluatedTemplateData]): Seq[EvaluatedTemplateData] = {
    val selectedParents = rankSelectionStrategy.parentSelectionByShareWithLocalCompetitions(0.5, evaluated)
    logger.debug(s"Selected parents: ${selectedParents.map(_.idShort).mkString(" , ")}")
    selectedParents
  }

  override def mutateParentPopulation(population: TPopulation): TPopulation = {
    // If we made sure that mutation Strategy ensures diversity than we need to perform extra mutations for duplications only in the case of cold start in the first iteration.
    val offspring = mutationStrategy.mutate(population) // duplicates are kind of a winners as well and that is unfair but we will eliminate it int the first iteration
    //TODO we can keep track on those who have already passed mutate function and see whether a new one is a duplicate or not.
    PopulationHelper.print(offspring, "Offspring population:")
    offspring
  }

  override def evaluatePopulation(population: TPopulation, workingDF: DataFrame): Seq[EvaluatedTemplateData] = {

    /* Template dimension depends on others dimensions and we need to get data from them first.
    This could be implemented in a custom hardcoded evaluator or with dependencies tree */
    //TODO  For how long we want to search for a hyperparameters? We can introduce HPSearchStepsPerGeneration parameter or we need to add logic that decides how often we need to evolve subdimensions
    logger.debug("Before evaluation of Template population we want to get best individuals from coevolutions we depend on. Checking HP coevolution...")
    if(hyperParamsEvDim.hallOfFame.isEmpty)
      hyperParamsEvDim.evolveFromLastPopulation(workingDF) // TODO consider stratified sample for first iteration or maybe for all iterations
    val bestHyperParametersField: HyperParametersField = hyperParamsEvDim.getBestFromHallOfFame

    if(problemType == MultiClassClassificationProblem) {
      evaluator.evaluateIndividuals(population, workingDF, bestHyperParametersField, problemType)
    }
    else {
      ???
    }
  }

  override def getBestFromPopulation(workingDF: DataFrame): EvaluatedTemplateData = {
    problemType match {
      case MultiClassClassificationProblem | BinaryClassificationProblem =>
        val defaultRegressionMetric = "f1"
        //TODO we might want to take evaluated population from dimension, Althoug it should be in the cache.
        getEvaluatedPopulation.sortWith(_.fitness.metricsMap(defaultRegressionMetric) > _.fitness.metricsMap(defaultRegressionMetric)).head
      case RegressionProblem =>
        // TODO maybe keep them in sorted heap?
        val defaultRegressionMetric = "rmse"
        getEvaluatedPopulation.sortWith(_.fitness.metricsMap(defaultRegressionMetric) < _.fitness.metricsMap(defaultRegressionMetric)).head
    }

  }

}
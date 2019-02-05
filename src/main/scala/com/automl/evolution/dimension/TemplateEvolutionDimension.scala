package com.automl.evolution.dimension
import akka.actor.{ActorRef, ActorSystem}
import com.automl.evolution.diversity.{DistinctDiversityStrategy, MisclassificationDistance}
import com.automl.evolution.evaluation.{TemplateNSLCEvaluator, TemplateSimpleEvaluator}
import com.automl.{EvaluatedTemplateData, Population, TPopulation}
import com.automl.evolution.mutation.{DepthDependentTemplateMutationStrategy, MutationProbabilities}
import com.automl.evolution.selection.RankSelectionStrategy
import com.automl.helper.{FitnessResult, PopulationHelper}
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.template.{TemplateMember, TemplateTree}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.param.Params
import org.apache.spark.sql.DataFrame

import scala.collection.mutable

/**
  *
  * @param evolveEveryGenerations is not used for now
  * @param problemType We need to take into account which models we can mutate into filtered out by problemType
  */
class TemplateEvolutionDimension(evolveEveryGenerations: Int = 1, problemType: ProblemType)(implicit val as: ActorSystem)
    extends EvolutionDimension[TPopulation, TemplateTree[TemplateMember], EvaluatedTemplateData]
    with LazyLogging{

  val distinctStrategy = new DistinctDiversityStrategy()
  val mutationStrategy = new DepthDependentTemplateMutationStrategy(distinctStrategy, problemType)
  val rankSelectionStrategy = new RankSelectionStrategy

  // Dependencies on other dimensions. Hardcoded for now. Should come from AutoML.runEvolution method parameters.
  val hyperParamsEvDim = new TemplateHyperParametersEvolutionDimension

  implicit val templatesEvaluationCache = mutable.Map[(TemplateTree[TemplateMember], Long), FitnessResult]()  // TODO make it faster with reference to value

  override def evolve(population: TPopulation, workingDF: DataFrame): TPopulation = {

    val evaluatedOriginalPopulation = evaluatePopulation(population, workingDF)

    evaluatedOriginalPopulation.printSortedByFitness()

    val selectedParents = selectParents(evaluatedOriginalPopulation)

    //    val selectedParents = rankSelectionStrategy.parentSelectionByShare(0.8, evaluatedOriginalPopulation)
    //Second phase: We are going to compute fitness functions and rank all the individuals.
    //Draw from these population with the probability distribution proportional to rank values.
    val bestTemplatesSelectedForMutation = selectedParents.map(_.template)

    //Problem: initial duplication of individuals. Are we allowed repetitions at all? For now lets keep it 100% diverse.
    // Duplications could be both in parents and losers.
    val losersIndividuals = evaluatedOriginalPopulation.diff(selectedParents)

    val duplicateTemplatesInLosersToMutate = if(losersIndividuals.distinct.size < losersIndividuals.size) {
      val distinctsInLosersTemplates = losersIndividuals.distinct
      losersIndividuals.diff(distinctsInLosersTemplates).map(_.template)
    } else Nil

    val populationForUpcomingMutation = new TPopulation(bestTemplatesSelectedForMutation ++ duplicateTemplatesInLosersToMutate, population.mutationProbabilities)
    val offspring = mutateParentPopulation(populationForUpcomingMutation)

    val mutantsEvaluationsForOffspringAndDuplicates = evaluatePopulation(offspring, workingDF)

    val evaluationResultsForNewExpandedGeneration = mutantsEvaluationsForOffspringAndDuplicates ++ evaluatedOriginalPopulation.distinct //losersEvaluations.distinct ++ bestEvaluations.distinct

    val survivedForNextGenerationEvaluatedTemplates = selectSurvived(population, evaluationResultsForNewExpandedGeneration)

    val evolvedPopulation = new TPopulation(survivedForNextGenerationEvaluatedTemplates.map(_.template), offspring.mutationProbabilities)

    // Do backpropagation of fitness. Evolve other dimensions by using new evaluations/best templates
//    hyperParamsEvDim.evolve()

    _population = evolvedPopulation
    evolvedPopulation
  }

  private def selectSurvived(population: TPopulation, evaluationResultsForNewExpandedGeneration: Seq[EvaluatedTemplateData]) = {
    rankSelectionStrategy.parentSelectionBySizeWithLocalCompetitions(population.size, evaluationResultsForNewExpandedGeneration)
  }

  override def selectParents(evaluated: Seq[EvaluatedTemplateData]): Seq[EvaluatedTemplateData] = {
    val selectedParents = rankSelectionStrategy.parentSelectionByShareWithLocalCompetitions(0.5, evaluated)
    selectedParents
  }

  override def mutateParentPopulation(population: TPopulation): TPopulation = {
    // If we made sure that mutation Strategy ensures diversity than we need to perform extra mutations for duplications only in the case of cold start in the first iteration.
    val offspring = mutationStrategy.mutate(population) // duplicates are kind of a winners as well and that is unfair but we will eliminate it int the first iteration
    //TODO we can keep track on those who have already passed mutate function and see whether a new one is a duplicate or not.
    logger.info("Offspring population:")
    PopulationHelper.print(offspring)
    offspring
  }

  override def evaluatePopulation(population: TPopulation, workingDF: DataFrame): Seq[EvaluatedTemplateData] = {
    //TODO implement Template pattern ( separate login into multiple functions and introduce them in EvolutionDimension)
    /* Template dimension depends on others dimensions and we need to get data from them first.
    This could be implemented in a custom hardcoded evaluator or with dependencies tree */
    //TODO  For how long we want to search for a hyperparameters? We can introduce HPSearchStepsPerGeneration parameter or we need to add logic that decides how often we need to evolve subdimensions
    val bestEvaluatedHyperParametersField: HyperParametersField = null
//    val bestEvaluatedHyperParametersField = hyperParamsEvDim.getBestFromPopulation(workingDF)// During next generation's call of this.evolve we will be able to get new/better individuals

    //TODO it is not generic to use MisclassificationDistance as we can be solving regression here as well
    if(problemType == MultiClassClassificationProblem)
      new TemplateNSLCEvaluator(new MisclassificationDistance)
        .evaluateIndividuals(population, workingDF, bestEvaluatedHyperParametersField, problemType)
    else {
      ???
    }
//    new TemplateSimpleEvaluator().evaluateIndividuals(population, workingDF, bestEvaluatedHyperParametersField, problemType)
  }

  override var _population: TPopulation = _

  override def getBestFromPopulation(workingDF: DataFrame): EvaluatedTemplateData = {
    problemType match {
      case MultiClassClassificationProblem | BinaryClassificationProblem =>
        val defaultRegressionMetric = "f1"
        evaluatePopulation(getPopulation, workingDF).sortWith(_.fitness.metricsMap(defaultRegressionMetric) > _.fitness.metricsMap(defaultRegressionMetric)).head
      case RegressionProblem =>
        // TODO maybe keep them in sorted heap?
        val defaultRegressionMetric = "rmse"
        evaluatePopulation(getPopulation, workingDF).sortWith(_.fitness.metricsMap(defaultRegressionMetric) < _.fitness.metricsMap(defaultRegressionMetric)).head
    }

  }

}
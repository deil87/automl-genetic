package com.automl.evolution.dimension
import akka.actor.{ActorRef, ActorSystem}
import com.automl.evolution.diversity.DistinctDiversityStrategy
import com.automl.{EvaluatedTemplateData, Population, TPopulationEvaluator, TPopulation}
import com.automl.evolution.mutation.{DepthDependentTemplateMutationStrategy, MutationProbabilities}
import com.automl.evolution.selection.RankSelectionStrategy
import com.automl.helper.{FitnessResult, PopulationHelper}
import com.automl.template.{TemplateMember, TemplateTree}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.ml.param.Params
import org.apache.spark.sql.DataFrame

import scala.collection.mutable

class TemplateEvolutionDimension(evolveEveryGenerations: Int = 1)(implicit val as: ActorSystem)
    extends EvolutionDimension[TPopulation, TemplateTree[TemplateMember], EvaluatedTemplateData]
    with LazyLogging{

  val distinctStrategy = new DistinctDiversityStrategy()
  val mutationStrategy = new DepthDependentTemplateMutationStrategy(distinctStrategy)
  val selectionStrategy = new RankSelectionStrategy

  // Dependencies on other dimensions. Hardcoded for now. Should come from AutoML.runEvolution method parameters.
  val hyperParamsEvDim = new TemplateHyperParametersEvolutionDimension

  implicit val templatesEvaluationCache = mutable.Map[(TemplateTree[TemplateMember], Long), FitnessResult]()  // TODO make it faster with reference to value

  override def evolve(population: TPopulation, workingDF: DataFrame): TPopulation = {

    val evaluatedOriginalPopulation = evaluatePopulation(population, workingDF)

    evaluatedOriginalPopulation.printSortedByFitness()

    val selectedParents = selectionStrategy.parentSelectionByShare(0.8, evaluatedOriginalPopulation)
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

    //For now number of individuals is bigger the original size of population. Need to shrink by selection.
    val survivedForNextGenerationEvaluatedTemplates = selectionStrategy.parentSelectionBySize(population.size, evaluationResultsForNewExpandedGeneration)

    val evolvedPopulation = new TPopulation(survivedForNextGenerationEvaluatedTemplates.map(_.template), offspring.mutationProbabilities)

    // Do backpropagation of fitness. Evolve other dimensions by using new evaluations/best templates
//    hyperParamsEvDim.evolve()

    _population = evolvedPopulation
    evolvedPopulation
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
    val bestEvaluatedHyperParametersField = hyperParamsEvDim.getBestFromPopulation(workingDF)// During next generation's call of this.evolve we will be able to get new/better individuals

    new TPopulationEvaluator().evaluateIndividuals(population, workingDF, bestEvaluatedHyperParametersField.field)
  }

  override var _population: TPopulation = _

  override def getBestFromPopulation(workingDF: DataFrame): EvaluatedTemplateData = {
    evaluatePopulation(getPopulation, workingDF).sortWith(_.fitness.fitnessError < _.fitness.fitnessError).head// TODO maybe keep them in sorted heap?
  }

}
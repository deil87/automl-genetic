package com.automl.evolution.dimension
import com.automl.{EvaluatedTemplateData, Population, PopulationEvaluator}
import com.automl.evolution.mutation.TemplateMutationStrategy
import com.automl.evolution.selection.RankSelectionStrategy
import com.automl.helper.FitnessResult
import com.automl.template.{TemplateMember, TemplateTree}
import org.apache.spark.ml.param.Params
import org.apache.spark.sql.DataFrame

import scala.collection.mutable

class TemplateEvolutionDimension extends EvolutionDimension {

  val mutationStrategy = new TemplateMutationStrategy //TODO to constructor
  val selectionStrategy = new RankSelectionStrategy

  // Dependencies on other dimensions. Hardcoded for now.
  val hyperParamsEvDim = new TemplateHyperParametersEvolutionDimension

  implicit val templatesEvaluationCache = mutable.Map[(TemplateTree[TemplateMember], Long), FitnessResult]()  // TODO make it faster with reference to value

  override def evolve(population: Population, workingDF: DataFrame): (Population, Option[EvaluatedTemplateData]) = {

    /* Template dimension depends on others dimensions and we need to get data from them first.
    This could be implemented in a custom hardcoded evaluator or with dependencies tree */
    val hyperParamsMap: Map[String, Seq[Params]] = hyperParamsEvDim.getBestPopulation()

    val evaluatedTemplatesData = evaluate(population, workingDF, hyperParamsMap)

    evaluatedTemplatesData.printSortedByFitness()

    //Second phase: We are going to compute fitness functions and rank all the individuals.
    //Draw from these population with the probability distribution proportional to rank values.
    val templatesForMutation = selectionStrategy.parentSelection(0.5, evaluatedTemplatesData).map(_.template)

    val offspring = mutationStrategy.mutate(new Population(templatesForMutation))

    val subjectsToSurvival = new Population(population.individuals ++ offspring.individuals)

    val evaluationResultsForAll = evaluate(subjectsToSurvival, workingDF, hyperParamsMap) //TODO we can evaluate only offspring here

    //Select 50% best of all the (individuals + offspring)
    val survivedForNextGenerationEvaluatedTemplates: Seq[EvaluatedTemplateData] = selectionStrategy.parentSelection(0.5, evaluationResultsForAll)
    val bestSurvivedEvaluatedTemplate = chooseBestIndividual(survivedForNextGenerationEvaluatedTemplates)

    val survivedTemplates = survivedForNextGenerationEvaluatedTemplates.map(_.template)
    //From seed population cause lots of duplication. Better select initialPopulationSize at once.
    val evolvedPopulation = Population.fromSeedPopulation(new Population(survivedTemplates)).withSize(population.size).build

    // Do backpropagation of fitness
    //hyperParamsEvDim.evolve()

    //Return evolved population
    (evolvedPopulation, bestSurvivedEvaluatedTemplate)
  }

  def evaluate(population: Population, workingDF: DataFrame, hyperParamsMap: Map[String, Seq[Params]]): Seq[EvaluatedTemplateData] =
    PopulationEvaluator.evaluateIndividuals(population, workingDF, hyperParamsMap)

  def applyMutation() = {}
  def select()= {}

  def chooseBestIndividual(evaluatedTemplates: Seq[EvaluatedTemplateData]): Option[EvaluatedTemplateData] = {

    evaluatedTemplates.sortWith(_.fitness.fitnessError < _.fitness.fitnessError).headOption // TODO maybe keep them in sorted heap?
  }

//  def dependencies = Set(new TemplateHyperParametersEvolutionDimension)

  override def getBestPopulation() = ???
}
package com.automl.evolution.dimension
import akka.actor.{ActorRef, ActorSelection, ActorSystem}
import com.automl.evolution.dimension.hparameter.{EvaluatedHyperParametersField, HyperParametersEvolutionDimension, HyperParametersField}
import com.automl.evolution.diversity.{DistinctDiversityStrategy, MisclassificationDistance}
import com.automl.evolution.evaluation.{EvaluationContextInfo, NeighboursFinder, TemplateNSLCEvaluator}
import com.automl.{ConfigProvider, EvaluatedTemplateData, EvaluatedTemplateDataDTOJsonProtocol, EvolutionProgressDTO, PaddedLogging}
import com.automl.evolution.mutation.{DepthDependentTemplateMutationStrategy, RandomTemplateMutationStrategy}
import com.automl.evolution.selection.RankSelectionStrategy
import com.automl.helper.{FitnessResult, PopulationHelper}
import com.automl.population.{GenericPopulationBuilder, TPopulation}
import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}
import com.automl.route.UpdateWebWithJson
import com.automl.template.{TemplateMember, TemplateTree}
import org.apache.spark.sql.DataFrame
import spray.json.DefaultJsonProtocol

import scala.collection.mutable
import scala.util.Random
import spray.json._

case class PopulationDTO( individuals: Seq[String], key: String = "population")

object PopulationDTOJsonProtocol extends DefaultJsonProtocol {
  implicit val format = jsonFormat2(PopulationDTO.apply)
}
/**
  *
  * @param evolveEveryNGenerations is not used for now
  * @param problemType We need to take into account which models we can mutate into filtered out by problemType
  */
class TemplateEvolutionDimension(initialPopulation: Option[TPopulation] = None, evolveEveryNGenerations: Int = 1, val problemType: ProblemType, seed: Long = new Random().nextLong())
    (implicit val as: ActorSystem, val logPaddingSize: Int)
    extends EvolutionDimension[TPopulation, TemplateTree[TemplateMember], EvaluatedTemplateData]
    with PaddedLogging{

  val webClientNotifier: ActorSelection = as.actorSelection("user/webClientNotifier")

  override def dimensionName: String = "TemplateDimension"

  val tdConfig = ConfigProvider.config.getConfig("evolution.templateDimension")

  lazy val evolutionDimensionLabel: String = tdConfig.getString("name")
  lazy val populationSize: Int = tdConfig.getInt("populationSize")

  val distinctStrategy = new DistinctDiversityStrategy()

//  val mutationStrategy = new DepthDependentTemplateMutationStrategy(distinctStrategy, problemType)(logPaddingSize + 4)
  val mutationStrategy = new RandomTemplateMutationStrategy(distinctStrategy, problemType, seed)(logPaddingSize + 4)
  val rankSelectionStrategy = new RankSelectionStrategy

  // Dependencies on other dimensions. Hardcoded for now. Should come from AutoML.runEvolution method parameters.
  val hyperParamsEvDim = if( ConfigProvider.config.getBoolean("evolution.hyperParameterDimension.enabled"))
    Some(new HyperParametersEvolutionDimension(this,problemType = problemType, seed = seed)(logPaddingSize + 8))
  else None

  override var _population: TPopulation = new TPopulation(Nil)

  val evaluator = if(problemType == MultiClassClassificationProblem) {
     new TemplateNSLCEvaluator(this, hyperParamsEvDim)(as, logPaddingSize + 4)
  } else throw new UnsupportedOperationException("Only MultiClassClassificationProblem is supported for now.")

  val neighboursFinder = new NeighboursFinder(new MisclassificationDistance)(as, logPaddingSize + 8)

   // TODO make it faster with reference to value
  override implicit val individualsEvaluationCache = mutable.Map[(TemplateTree[TemplateMember], Long), FitnessResult]()

  // TODO generalize and make dependent dimensions to be used in the key automatically. maybe just use string with tree respresentation inside
  implicit val individualsEvaluationCacheExtended = mutable.Map[(TemplateTree[TemplateMember], Option[HyperParametersField], Long, Int), FitnessResult]()

  override val hallOfFame: mutable.PriorityQueue[EvaluatedTemplateData] = collection.mutable.PriorityQueue[EvaluatedTemplateData]()


  override def getInitialPopulationFromMetaDB: TPopulation = ??? //new TPopulation(metaDB.getPopulationOfTemplates)

  override def getInitialColdStartPopulation: TPopulation = {
    initialPopulation.map{population =>
      GenericPopulationBuilder.fromSeedPopulation(population)
        .withProblemType(problemType)
        .withSize(populationSize)
        .build
    }.getOrElse(throw new IllegalStateException("Initial population was not specified")) // TODO we should provide default populatin here
  }

  //Almost generalisable. Need to specify type that is common to _.template and _.field
  override def getBestFromHallOfFame: TemplateTree[TemplateMember] = hallOfFame.headOption.map(_.template).getOrElse{getInitialPopulation.individuals.head}

  override def showCurrentPopulation(): Unit = {
    if(getEvaluatedPopulation.nonEmpty)
      debug(PopulationHelper.renderEvaluatedIndividuals(getEvaluatedPopulation, "Last evaluated population from previous generation"))
    else PopulationHelper.print(getPopulation, "Current population without evaluations")
  }

  var skipEvolutionCountDown: Int = evolveEveryNGenerations - 1

  override def evolve(population: TPopulation, workingDF: DataFrame, evaluationContextInfo: EvaluationContextInfo): TPopulation = {

    //TODO Control should be outside of Dimension concept
    //Maybe it is better to set up flag for corner case like 'always execute'
    if(skipEvolutionCountDown > 0) {
      debug(s"SKIPPING evolution. Next evolution in $skipEvolutionCountDown attempts")
      skipEvolutionCountDown -= 1
      return getPopulation
    }
    skipEvolutionCountDown = evolveEveryNGenerations - 1

    showCurrentPopulation()
    //TODO should we retrain whole population once workingDF increased?
    val evaluatedPopulation = getLastEvaluatedPopulation(workingDF, evaluationContextInfo)

    import PopulationDTOJsonProtocol._
    val populationDTO = PopulationDTO(evaluatedPopulation.map(_.id)).toJson
    webClientNotifier ! UpdateWebWithJson(populationDTO.prettyPrint)

    //Need to decide where selecting neighbours should go. To evaluation or selection or to its own phase.
    debug("Finding neighbours for NSLC algorithm:")
    val evaluatedOriginalPopulationWithNeighbours = neighboursFinder.findNeighbours(evaluatedPopulation, evaluatedPopulation, population.size, problemType)

    debug("Selecting parents:")
    val selectedParents = selectParents(evaluatedOriginalPopulationWithNeighbours)

    //Problem: initial ot on the way duplication of individuals. Are we allowed repetitions at all? For now lets keep try to force diversity on mutation phase.
    val populationForUpcomingMutation = extractIndividualsFromEvaluatedIndividuals(selectedParents)

    debug("Mutating parents:")
    val offspring = mutateParentPopulation(populationForUpcomingMutation, new TPopulation(hallOfFame.map(_.template).toSeq))

    debug("Evaluating offspring:")
    val evaluatedOffspring = evaluatePopulation(offspring, workingDF, evaluationContextInfo)

//    debug("Updating hallOfFame:") // TODO maybe we don't need to update it here as we did it during evaluations in Evaluator
//    updateHallOfFame(evaluatedOffspring)

    val evaluatedOffspringWithNeighbours = neighboursFinder.findNeighbours(evaluatedOffspring, evaluatedOffspring ++ evaluatedOriginalPopulationWithNeighbours, population.size, problemType)

    val evaluationResultsForNewExpandedGeneration = evaluatedOffspringWithNeighbours ++ evaluatedOriginalPopulationWithNeighbours

    debug("Selecting survivals:")
    val survivedForNextGenerationEvaluatedTemplates = selectSurvived(population.size, evaluationResultsForNewExpandedGeneration)

    _evaluatedEvolvedPopulation = survivedForNextGenerationEvaluatedTemplates.sortWith(_.betterThan(_))

    val evolvedPopulation = new TPopulation(survivedForNextGenerationEvaluatedTemplates.map(_.template))

    // Do backpropagation of fitness. Evolve other dimensions by using new evaluations/best templates
    debug(s"Evolving dependent coevolution hyperParamsEvDim: enabled = ${hyperParamsEvDim.isDefined}")
    hyperParamsEvDim.foreach(_.evolveFromLastPopulation(workingDF, evaluationContextInfo))

    _population = evolvedPopulation
    evolvedPopulation
  }

  override def selectSurvived(populationSize: Int, evaluationResultsForNewExpandedGeneration: Seq[EvaluatedTemplateData]) = {
    val res = rankSelectionStrategy.selectionBySizeWithLocalCompetitions(populationSize, evaluationResultsForNewExpandedGeneration)
    info("selectSurvived is finished")
    res
  }


  override def extractIndividualsFromEvaluatedIndividuals(evaluatedIndividuals: Seq[EvaluatedTemplateData]): TPopulation = {
    new TPopulation(evaluatedIndividuals.map(_.template))
  }

  override def selectParents(evaluated: Seq[EvaluatedTemplateData]): Seq[EvaluatedTemplateData] = {
    val selectedParents = rankSelectionStrategy.parentSelectionByShareWithLocalCompetitions(0.5, evaluated)
    debug(s"Selected parents: ${selectedParents.map(_.idShort).mkString(" , ")}")
    selectedParents
  }

  override def mutateParentPopulation(population: TPopulation, notToIntersectWith: TPopulation): TPopulation = {
    // If we made sure that mutation Strategy ensures diversity than we need to perform extra mutations for duplications only in the case of cold start in the first iteration.
    val offspring = mutationStrategy.mutate(population, notToIntersectWith) // duplicates are kind of a winners as well and that is unfair but we will eliminate it int the first iteration
    //TODO we can keep track on those who have already passed mutate function and see whether a new one is a duplicate or not.
    PopulationHelper.print(offspring, "Offspring population:")
    offspring
  }

  override def evaluatePopulation(population: TPopulation, workingDF: DataFrame, evaluationContextInfo: EvaluationContextInfo): Seq[EvaluatedTemplateData] = {

    if(problemType == MultiClassClassificationProblem) {
      evaluator.evaluateIndividuals(population, workingDF, problemType, evaluationContextInfo, seed)
    }
    else {
      ???
    }
  }
}
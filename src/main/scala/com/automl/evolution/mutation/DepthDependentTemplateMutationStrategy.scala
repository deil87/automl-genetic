package com.automl.evolution.mutation

import com.automl.evolution.dimension.hparameter.TemplateHyperParametersEvolutionDimension
import com.automl.{ConfigProvider, PaddedLogging}
import com.automl.evolution.diversity.DiversityStrategy
import com.automl.evolution.selection.{RankBasedSelectionProbabilityAssigner, RouletteWheel}
import com.automl.population.{HPPopulation, TPopulation}
import com.automl.problemtype.ProblemType
import com.automl.template.ensemble.EnsemblingModelMember
import com.automl.template.simple.SimpleModelMember
import com.automl.template.simple.SimpleModelMember.poolOfSimpleModels
import com.automl.template.{LeafTemplate, NodeTemplate, TemplateMember, TemplateTree}
import com.typesafe.config.ConfigFactory
import com.typesafe.scalalogging.LazyLogging
import scalaz.Lens

import scala.collection.immutable
import scala.util.Random
import scala.collection.JavaConverters._

//Operate on a population-wide scale, so mutation function is somewhere inside in the dungeons
/**
  *   Main idea of this strategy is to calculate probability for the node to be mutated during current mutation phase
  *   based on the depth of a particular node in the whole template's tree - the deeper into the tree we get the more likely we will mutate.
  *    There are some options also for how we increase complexity/add ensembling nodes. Probably we can add special strategy for
  *         1) We can mutate only ensembling node by adding  children to it (base model nodes or another ensembling nodes).
  *           In case the template is a single base model we can mutate it to another base model or to ensemble node.
  *         2) Or we can mutate any node to anything.
  * @param diversityStrategy
  */
class DepthDependentTemplateMutationStrategy(diversityStrategy: DiversityStrategy, problemType: ProblemType)(implicit val logPaddingSize: Int) extends PaddedLogging {

  val tdConfig = ConfigProvider.config.getConfig("evolution.templateDimension")

  lazy val maxEnsembleDepth: Int = tdConfig.getInt("maxEnsembleDepth")
  lazy val maxNumberOfMutationAttempts: Int = tdConfig.getInt("maxNumberOfMutationAttempts")
  lazy val pivotBetweenStructureAndHPMutations: Double = tdConfig.getDouble("pivotBetweenStructureAndHPMutations")

  val hPMutationStrategy = new HPMutationStrategy()(4)

  /**
    *
    * @param population Among other things it contains population-wide mutation probabilities
    * @return
    */
  def mutate(population: TPopulation, populationNotToIntersectWith: TPopulation = null): TPopulation = {

    info(s"Starting new mutation phase for the population...")

    // TODO Warning. It is more about deduplication edge case of preserving diversity.  Better to think about something more efficient.
    //TODO lets try without preserving diversity strategy. Hopefully we won't be introducing many during single mutation of population.
    // If we restrinct duplicates or similar indeviduals than we are risking to not searching around the optimal ensemble.
    // We would have to compensate this restriction with more search-time for best individuals
    ////val res = diversityStrategy.apply(population, mutateIndividual)
    val mutatedIndividuals = population.individuals.foldLeft(List.empty[TemplateTree[TemplateMember]])((res, next) => {
      var attempts = 1
      var newMutant: TemplateTree[TemplateMember] = null
      do {
        if(attempts > 1) info(s"\t\t Attempt number ${attempts - 1} was unsuccessful. Max. number of attemts from config = $maxNumberOfMutationAttempts")
        newMutant = mutateIndividual(next)
        attempts += 1
      } while ( (population.individuals ++ res ++ populationNotToIntersectWith.individuals).contains(newMutant) && attempts < maxNumberOfMutationAttempts) // Not to overlap with itself, with recently mutated new individuals and custom individuals from `notToIntersectWith`
      info(s"Mutation of the ${next.id}:${next.member.name.take(5)} individual took ${attempts-1} attempts.")
      require(attempts != maxNumberOfMutationAttempts, s"Too many attempts to mutate ${next.id} with DepthDependentTemplateMutationStrategy.")
      val list = newMutant +: res
      list
    })
    new TPopulation(mutatedIndividuals)
  }

  // We should check after mutation whether we have already encountered individuals like this before. Compare with the cache and current population.
  // After a mutation phase over whole population (as it is happenning sequentially) we need to make sure that there are no duplicates...
  // or we can do it along the way by checking cache and already processed part of population.
  def mutateIndividual(individual: TemplateTree[TemplateMember]):TemplateTree[TemplateMember] = {


    def getRandomEnsemblingMember = EnsemblingModelMember.randomMember

    def getRandomBaseMemberBasedOnProblemType = SimpleModelMember.randomMember(problemType)

    def getRandomBaseMemberWithExclusion(exclude: Seq[SimpleModelMember]): Option[TemplateMember] =
      SimpleModelMember.randomMemberWithExclusion(problemType, exclude)

    /**
      * @param depth
      * @return level of the template's tree to mutate. Indexing from the top. Root level is 0 level.
      */
    def chooseLevelOfMutationBasedOnDepth( depth: Int): Int = {

      val contenders = (0 until depth).toList
      val probabilityAssigner = new RankBasedSelectionProbabilityAssigner[Int].assign(contenders)
      val selector = new RouletteWheel[Int](probabilityAssigner)
      val levelOfMutation = selector.getNext._1

      levelOfMutation
    }

    def mutateLeafToNode(lt: LeafTemplate[TemplateMember]) = {
      val numberOfNewChildren = new Random().nextInt(2) + 1
      val randomEnsemblingMember = getRandomEnsemblingMember
      val oneForOriginalTemplate = 1
      info(s"\t\t Mutation happened from leaf node $lt to ensembling of ${numberOfNewChildren + oneForOriginalTemplate} submembers - $randomEnsemblingMember , causing increasing of complexity.")
      NodeTemplate(randomEnsemblingMember, Seq(lt) ++ (0 until numberOfNewChildren).map(_ => LeafTemplate(getRandomBaseMemberBasedOnProblemType)))
    }

    // We should perform one action of mutation per template. Somewhere in the tree. TODO probably it is more efficient to store level in nodes
    def traverseAndMutate(individual: TemplateTree[TemplateMember], currentLevel: Int,
                          targetLevelOfMutation: Int): TemplateTree[TemplateMember] = individual match {
      case lt@LeafTemplate(member) =>
        if (targetLevelOfMutation == currentLevel) {
          val rg = new Random()
          val pivotStructureVsHPs = rg.nextDouble()
          if (pivotStructureVsHPs > pivotBetweenStructureAndHPMutations) {
            mutateHPMap(lt)
          } else {
            mutateStructureInCaseOfLeafNode(getRandomBaseMemberWithExclusion _, mutateLeafToNode _, currentLevel, lt, member, rg)
          }
        } else { // TODO how we end up here? Should not get here.
          info("Dead end. Lost an opportunity to mutate. Should not happen.")
          lt
        }

      case nt@NodeTemplate(ensemblingMember, subMembers) =>
        if(targetLevelOfMutation == currentLevel) {
          val pivot = new Random().nextDouble()
          if(currentLevel < maxEnsembleDepth - 1 && pivot > 0.7) {
            info("\t\t Mutate ensembling node by adding new ensembling node to its submembers")
            //TODO not just add ensembling node but replace some of the sub members
            val numberOfNewChildren = new Random().nextInt(2) + 1
            val randomEnsemblingMember = getRandomEnsemblingMember
            NodeTemplate(ensemblingMember, subMembers :+ NodeTemplate(randomEnsemblingMember,  (0 until numberOfNewChildren).map(_ => LeafTemplate(getRandomBaseMemberBasedOnProblemType))))
          }
          else {
            info("\t\t Mutate ensembling node by adding new leaf template to its submembers")
            NodeTemplate(ensemblingMember, subMembers :+ LeafTemplate(getRandomBaseMemberBasedOnProblemType))
          }
        } else {
          val randSubmember = subMembers.randElement
          //TODO we are changing order here and it might change representation that we will be using as a key for caching
          NodeTemplate(ensemblingMember, subMembers.diff(Seq(randSubmember)) :+ traverseAndMutate(randSubmember, currentLevel + 1, targetLevelOfMutation))
        }

    }

    val targetLevelOfMutation = chooseLevelOfMutationBasedOnDepth(individual.height)

    traverseAndMutate(individual, 0, targetLevelOfMutation)
  }

  private def mutateStructureInCaseOfLeafNode(getRandomBaseMemberWithExclusion: Seq[SimpleModelMember] => Option[TemplateMember], mutateLeafToNode: LeafTemplate[TemplateMember] => NodeTemplate[TemplateMember], currentLevel: Int, lt: LeafTemplate[TemplateMember], member: TemplateMember, rg: Random) = {
    if (rg.nextDouble() > 0.8 && currentLevel < maxEnsembleDepth - 1) { // Note currentLevel is zero-based
      mutateLeafToNode(lt.asInstanceOf[LeafTemplate[SimpleModelMember]])
    } else {
      val randomBaseMemberBasedOnProblemType = getRandomBaseMemberWithExclusion(Seq(member.asInstanceOf[SimpleModelMember])).asInstanceOf[Option[SimpleModelMember]]
      // TODO rewrite so that we don't need to cast member to SimpleModelMember
      randomBaseMemberBasedOnProblemType match {
        case Some(randomBaseMember) =>
          info(s"\t\t Mutation happened from leaf node $lt to another leaf node $randomBaseMemberBasedOnProblemType")
          LeafTemplate(randomBaseMember)
        case None =>
          if (currentLevel < maxEnsembleDepth - 1) {
            mutateLeafToNode(lt.asInstanceOf[LeafTemplate[SimpleModelMember]])
          } else {
            info("Dead end. Lost an opportunity to mutate. Just mutate HyperParameters field")
            lt
          }
      }
    }
  }

  private def mutateHPMap(lt: TemplateTree[TemplateMember]): TemplateTree[TemplateMember] = {
    val hpPopulationToMutate = new HPPopulation(lt.internalHyperParamsMap.toSeq)
    lt.internalHyperParamsMap = hPMutationStrategy.mutate(hpPopulationToMutate).individuals.headOption
    lt
  }

  def mutateNTimes(population: TPopulation, times: Int): TPopulation = {
    def mutationHelper(population: TPopulation, countdown: Int): TPopulation = {
      if(countdown == 0) population
      else {
        val mutatedPopulation = mutate(population)
        mutatedPopulation.individuals.foreach(template => print(template.render))
        mutationHelper(mutatedPopulation, countdown - 1)
      }
    }

    mutationHelper(population, times)
  }
}

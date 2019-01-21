package com.automl.evolution.mutation

import com.automl.{Population, TPopulation}
import com.automl.evolution.diversity.DiversityStrategy
import com.automl.evolution.selection.{RankBasedSelectionProbabilityAssigner, RouletteWheel}
import com.automl.template.ensemble.EnsemblingModelMember
import com.automl.template.simple.SimpleModelMember
import com.automl.template.{LeafTemplate, NodeTemplate, TemplateMember, TemplateTree}
import com.typesafe.scalalogging.LazyLogging

import scala.collection.immutable
import scala.util.Random

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
class DepthDependentTemplateMutationStrategy(diversityStrategy: DiversityStrategy) extends LazyLogging {

  /**
    *
    * @param population Among other things it contains population-wide mutation probabilities
    * @return
    */
  def mutate(population: TPopulation): TPopulation = {

    logger.info(s"\n\nStarting new mutation phase for the population...")

    def mutateIndividual(individual: TemplateTree[TemplateMember]) = {


      def getRandomEnsemblingMember = EnsemblingModelMember.poolOfEnsemblingModels.toSeq.randElement

      def getRandomBaseMember: TemplateMember = SimpleModelMember.poolOfSimpleModels.randElement

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

      // We should perform one action of mutation per template. Somewhere in the tree. TODO probably it is more efficient to store level in nodes
      def traverseAndMutate(individual: TemplateTree[TemplateMember], currentLevel: Int,
                            targetLevelOfMutation: Int): TemplateTree[TemplateMember] = individual match {
        case lt@LeafTemplate(_) =>
          if(targetLevelOfMutation == currentLevel) {
            val pivot = new Random().nextDouble()
            if(pivot > 0.8) { // ToDO with `doWithProbability` helper method. DoOtherwise?
              logger.info("Mutate leaf node to ensembling one causing encreasing of complexity")
              val numberOfNewChildren = new Random().nextInt(3)
              NodeTemplate(getRandomEnsemblingMember, Seq(lt) ++ (1 to numberOfNewChildren).map(_ => LeafTemplate(getRandomBaseMember)))
            } else {
              logger.info("Mutate leaf node to another leaf node")
              LeafTemplate(getRandomBaseMember)
            }
          } else
            {
              logger.info("Dead end. Lost an opportunity to mutate.")
              lt
            }

        case nt@NodeTemplate(ensemblingMember, subMembers) =>
          if(targetLevelOfMutation == currentLevel) {
            val pivot = new Random().nextDouble()
            if(pivot > 0.8) {
              logger.info("Mutate ensembling node by adding new leaf template to its submembers")
              NodeTemplate(ensemblingMember, subMembers :+ LeafTemplate(getRandomBaseMember))
            } else {
              logger.info("Mutate ensembling node by adding new ensembling node to its submembers")
              //TODO not just add ensembling node but replace some of the sub members
              val numberOfNewChildren = new Random().nextInt(3)
              NodeTemplate(ensemblingMember, subMembers :+ NodeTemplate(getRandomEnsemblingMember,  (1 to numberOfNewChildren).map(_ => LeafTemplate(getRandomBaseMember))))
            }

          } else {
            traverseAndMutate(subMembers.randElement, currentLevel + 1, targetLevelOfMutation)
          }

      }

      val targetLevelOfMutation = chooseLevelOfMutationBasedOnDepth(individual.height)

      traverseAndMutate(individual, 0, targetLevelOfMutation)
    }

    // TODO Warning. It is more about deduplication edge case of preserving diversity.  Better to think about something more efficient.
    //TODO lets try without preserving diversity strategy. Hopefully we won't be introducing many during single mutation of population.
    // If we restrinct duplicates or similar indeviduals than we are risking to not searching around the optimal ensemble.
    // We would have to compensate this restriction with more search-time for best individuals
    ////val res = diversityStrategy.apply(population, mutateIndividual)
    new TPopulation(population.individuals.map(mutateIndividual), population.mutationProbabilities)

  }
}

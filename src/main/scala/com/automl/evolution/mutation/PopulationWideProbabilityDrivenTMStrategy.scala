package com.automl.evolution.mutation

import com.automl.Population
import com.automl.evolution.diversity.DiversityStrategy
import com.automl.template.ensemble.EnsemblingModelMember
import com.automl.template.simple.SimpleModelMember
import com.automl.template.{LeafTemplate, NodeTemplate, TemplateMember, TemplateTree}
import com.typesafe.scalalogging.LazyLogging

import scala.util.Random


/**
  *   Template Mutation Strategy.
  *
  *   Main idea of this strategy is to calculate probability for the node to be mutated during current mutation phase
  *   based on the population wide probabilities.
  *    There are some options also for how we increase complexity/add ensembling nodes. Probably we can add special strategy for
  *         1) We can mutate only ensembling node by adding  children to it (base model nodes or another ensembling nodes).
  *           In case the template is a single base model we can mutate it to another base model or to ensemble node.
  *         2) Or we can mutate any node to anything.
  * @param diversityStrategy
  */
class PopulationWideProbabilityDrivenTMStrategy(diversityStrategy: DiversityStrategy) extends LazyLogging {

  /**
    *
    * @param population Among other things it contains population-wide mutation probabilities
    * @return
    */
  def mutate(population: Population): Population = {

    logger.info(s"Starting new mutation phase for the population...")

    def mutateIndividual(individual: TemplateTree[TemplateMember]) = {


      def getRandomEnsemblingMember = EnsemblingModelMember.poolOfEnsemblingModels.toSeq.randElement

      def getRandomBaseMember: TemplateMember = SimpleModelMember.poolOfSimpleModels.randElement

      /**
        * We can learn over time and datasets about which level to mutate. #metalearning
        * @param probabilities
        * @param drawnModel
        * @param depth
        * @return level of the template's tree to mutate
        */
      def chooseLevelOfMutationBasedOnPopulationWideMutationProbabilities(probabilities: MutationProbabilities, drawnModel: TemplateMember, depth: Int): Int = {

        val levelOfMutation = if(probabilities.noveltySection.contains(drawnModel)) {
          //Mutate root element because we have drawn a member based on Population-wide probabilities (from one level up from root node of the individual)
          0  // 0 - root level
        } else {
          new Random().nextInt(depth-1) + 1
        }
        levelOfMutation
      }

      // We should perform one action of mutation per template. Somewhere in the tree.
      def traverseAndMutate(individual: TemplateTree[TemplateMember],  mutationProbabilities: MutationProbabilities, nextModel: TemplateMember, targetLevelOfMutation: Int): TemplateTree[TemplateMember] = individual match {
        case lt@LeafTemplate(_) =>

          //  Should we change structure in leaf node? Or mutate into ensembling node.
          // And we are not mutating leaf nodes. directly. Being in Ensemble node we can decide to add Ensemble children.
          val decidedToMutateLeafIntoEnsemble = false
          if(decidedToMutateLeafIntoEnsemble)
              NodeTemplate(getRandomEnsemblingMember, Seq(lt) ++ (0 to ???).map(_ => LeafTemplate(getRandomBaseMember)))
          else {
            logger.info("Attempt to mutate leaf node. For now we are not doing this. We have chosen to increase depth of template's trees by adding ensemble node as a child to ensemble node being under mutation")
            lt
          }

        case nt@NodeTemplate(ensemblingMember, subMembers) =>

          //TODO Check how well we have explored space. Should be automatically controlled by MutationProbabilities on this Ensemple node.

          if(targetLevelOfMutation == 0) {
            val mutatedPartOfTree = nextModel match {
              case ens: EnsemblingModelMember => NodeTemplate(nextModel, Seq(nt))
              case ens: SimpleModelMember => LeafTemplate(nextModel)
            }
            mutationProbabilities.moveToExperienceSection(nextModel)
            mutatedPartOfTree
          } else {
            // We can take into considerations Probabilities objects from different levels of tree. For now lets use only population-wide probabilities.
            //we don't want to mutate all the submembers. Need to choose only one
            val (randomSubMember, restSubMembers) = subMembers.extractElem
            NodeTemplate(ensemblingMember, restSubMembers :+ traverseAndMutate(randomSubMember, mutationProbabilities, nextModel, targetLevelOfMutation - 1))
          }

      }

      val mutationProbabilitiesPopWide: MutationProbabilities = population.mutationProbabilities
      val (nextModel, _) = mutationProbabilitiesPopWide.getNextMember // TODO we need somehow update novelty section before we will get feedback from evaluation process. Otherwise all the individuals will mutate from novelty sections
      //TODO we can probably move it to the experience section straight away....and later update it with experience and recalculate all distribution for Experience section.
      // We reestimate probabilities on the generation level - at the end of every generation.

      val targetLevelOfMutation = chooseLevelOfMutationBasedOnPopulationWideMutationProbabilities(mutationProbabilitiesPopWide, nextModel, individual.height)

      traverseAndMutate(individual, mutationProbabilitiesPopWide, nextModel, targetLevelOfMutation)
    }

    //TODO lets try without preserving diversity strategy. Hopefully we won't be introducing many during single mutation of population.
    // If we restrinct duplicates or similar indeviduals than we are risking to not searching around the optimal ensemble.
    // We would have to compensate this restriction with more search-time for best individuals
    ////val res = diversityStrategy.apply(population, mutateIndividual)
    new Population(population.individuals.map(mutateIndividual), population.mutationProbabilities)

  }
}

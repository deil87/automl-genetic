package com.automl.evolution.mutation

import com.automl.Population
import com.automl.evolution.diversity.DiversityStrategy
import com.automl.template.ensemble.EnsemblingModelMember
import com.automl.template.simple.SimpleModelMember
import com.automl.template.{LeafTemplate, NodeTemplate, TemplateMember, TemplateTree}
import com.typesafe.scalalogging.LazyLogging

import scala.util.Random

//Operate on a population-wide scale, so mutation function is somewhere inside in the dungeons

class TemplateMutationStrategy(diversityStrategy: DiversityStrategy) extends LazyLogging {

  /**
    *
    * @param population Among other things it contains population-wide mutation probabilities
    * @return
    */
  def mutate(population: Population): Population = {

    logger.info(s"\n\nStarting new mutation phase for the population...")

    def mutate(individual: TemplateTree[TemplateMember]) = {


      def getRandomEnsemblingMember = EnsemblingModelMember.poolOfEnsemblingModels.toSeq.randElement

      def getRandomBaseMember: TemplateMember = SimpleModelMember.poolOfSimpleModels.randElement

      def chooseMutationTypeBasedOnPopulationWideMutationProbabilities = {
        val probabilities: MutationProbabilities = population.mutationProbabilities


      }

      // We should perform one action of mutation per template. Somewhere in the tree.
      def traverseAndMutate(individual: TemplateTree[TemplateMember]): TemplateTree[TemplateMember] = individual match {
        case lt@LeafTemplate(_) =>

          // Here we can start mutate other dimensions. Do not change structure. Or mutate into ensembling node.
          // And we are not mutating leaf nodes. directly. Being in Ensemble node we can decide to add Ensemble children.
              NodeTemplate(getRandomEnsemblingMember, Seq(lt) ++ (0 to ???).map(_ => LeafTemplate(getRandomBaseMember)))


        case nt@NodeTemplate(ensemblingMember, subMembers) =>

          //TODO Check how well we have explored space. Should be automatically controlled by MutationProbabilities on this Ensemple node.


          // Here we can start mutate other dimensions. Do not change structure.

          NodeTemplate(ensemblingMember, subMembers.map(traverseAndMutate(_)))

      }

      traverseAndMutate(individual)
    }

    val res = diversityStrategy.apply(population, mutate)
    res
  }
}

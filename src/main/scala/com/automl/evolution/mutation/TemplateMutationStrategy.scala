package com.automl.evolution.mutation

import com.automl.Population
import com.automl.helper.MutationProbabilities
import com.automl.template.ensemble.EnsemblingMember
import com.automl.template.simple.SimpleModelMember
import com.automl.template.{LeafTemplate, NodeTemplate, TemplateMember, TemplateTree}
import com.typesafe.scalalogging.LazyLogging

import scala.util.Random

class TemplateMutationStrategy extends LazyLogging {

  def mutate(population: Population): Population = {

    logger.info(s"\n\nStarting new mutation phase...")

    def mutate(individual: TemplateTree[TemplateMember]) = {

      val individualsTreeHeight = individual.height
      val initialProbability, probStep: Double = 1.0 / individualsTreeHeight
      logger.info(s"Initial probability: $initialProbability, probStep: $probStep")
      val mutationProbabilities = MutationProbabilities(initialProbability)

      def getRandomEnsemblingMember = EnsemblingMember.poolOfEnsemblingModels.toSeq.randElement

      def getRandomBaseMember: TemplateMember = SimpleModelMember.poolOfSimpleModels.randElement

      val structOrMemberThreshold = 0.5

      def traverseAndMutate(individual: TemplateTree[TemplateMember], mutProbs: MutationProbabilities): TemplateTree[TemplateMember] = individual match {
        case lt@LeafTemplate(_) =>
          if (structOrMemberThreshold >= Random.nextDouble()) {
            if (mutProbs.structureProb >= Random.nextDouble()) {

              val numberOfNewBaseModels = Random.nextInt(4) + 1 // TODO parameter?

              val newMember = NodeTemplate(getRandomEnsemblingMember, Seq(lt) ++ (0 to numberOfNewBaseModels).map(_ => LeafTemplate(getRandomBaseMember)))
              logger.info(s"\nStructural mutation happened for $lt --> $newMember")
              newMember
            }
            else if (mutProbs.memberProb >= Random.nextDouble()) {
              val newMember: TemplateMember = getRandomBaseMember
              logger.info(s"\nMember mutation happened for $lt --> $newMember")
              LeafTemplate(newMember)
            }
            else lt
          }
          else lt

        case nt@NodeTemplate(ensemblingMember, subMembers) =>
          val updatedMutationProbs = mutProbs.increaseAllBy(probStep)

          NodeTemplate(ensemblingMember, subMembers.map(traverseAndMutate(_, updatedMutationProbs)))

      }

      traverseAndMutate(individual, mutationProbabilities)
    }

    new Population(population.individuals map mutate)
  }
}

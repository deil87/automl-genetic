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

  def mutate(population: Population): Population = {

    logger.info(s"\n\nStarting new mutation phase for the population...")

    def mutate(individual: TemplateTree[TemplateMember]) = {



      def getRandomEnsemblingMember = EnsemblingModelMember.poolOfEnsemblingModels.toSeq.randElement

      def getRandomBaseMember: TemplateMember = SimpleModelMember.poolOfSimpleModels.randElement

      val hierarhy2memberMutationthreshold = 0.8 // When we have explored this fraction of combinations on particular # of levels then we can allow for hierarchical mutations.

      def probOfHierarhyMutation(individual: TemplateTree[TemplateMember]) = {
        val individualHeight = individual.height

        if(individualHeight == 1) {

        }

      }


      def traverseAndMutate(individual: TemplateTree[TemplateMember]): TemplateTree[TemplateMember] = individual match {
        case lt@LeafTemplate(_) =>

              NodeTemplate(getRandomEnsemblingMember, Seq(lt) ++ (0 to ???).map(_ => LeafTemplate(getRandomBaseMember)))
              //newMember
//
//              val newMember: TemplateMember = getRandomBaseMember
//              LeafTemplate(???/*newMember*/)

        case nt@NodeTemplate(ensemblingMember, subMembers) =>
          val updatedMutationProbs = ??? // mutProbs.increaseAllBy(probStep)

          NodeTemplate(ensemblingMember, subMembers.map(traverseAndMutate(_)))

      }

      traverseAndMutate(individual)
    }

    val res = diversityStrategy.apply(population, mutate)
    res
  }
}

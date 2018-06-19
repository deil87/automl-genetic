package com.automl.evolution.mutation

import com.automl.helper.Probability
import com.automl.template.TemplateMember
import com.automl.template.ensemble.EnsemblingModelMember
import com.automl.template.simple.SimpleModelMember

import scala.collection.mutable

case class MutationProbabilities(complexityFactor: Double = 0.8,
                                 simpleModels: Set[SimpleModelMember] = SimpleModelMember.poolOfSimpleModels.toSet,
                                 ensembleModels: Set[EnsemblingModelMember] = EnsemblingModelMember.poolOfEnsemblingModels
                                ) {
  val cardinality: Int = simpleModels.size + ensembleModels.size

  def simpleModelsDefaults = mutable.Map(
    simpleModels.toSeq.map(model => {
      val simpleModelDefaultProb = Probability(complexityFactor / simpleModels.size)
      (model, simpleModelDefaultProb)
    }):_*
  )

  def ensembleModelsDefaults = mutable.Map(
    ensembleModels.toSeq.map(model => {
      val complexModelDefaultProb = Probability((1 - complexityFactor) / ensembleModels.size)
      (model, complexModelDefaultProb)
    }):_*
  )

  val noveltySection: mutable.Map[TemplateMember, Probability] = defaultProbabilities
  val experienceSection: mutable.Map[TemplateMember, ProbabilityContext] = mutable.Map.empty

  def probabilities: mutable.Map[TemplateMember, Probability] = noveltySection ++ experienceSection.map{case (t, pc) => (t, pc.finalProb)}

  def defaultProbabilities: mutable.Map[TemplateMember, Probability] = simpleModelsDefaults ++ ensembleModelsDefaults

  val maxExperiencedScore: Double = 0

  def update(updatedMember: TemplateMember, score: Double): Unit = {
    noveltySection.remove(updatedMember).map { noveltyPob =>
      //Case when there were no experience before
//      val totalProbFraction = experienceSection.map(_._2.value).sum

      //At first we just add new credit item. We move amount of probability from novelty to experience section. Then we will recalculate all probabilities.
      experienceSection += ((updatedMember, ProbabilityContext(noveltyPob, Seq(score))))
      val allScores = experienceSection.flatMap(_._2.scores) //TODO implement as incremental avg count?
      val amountOfAvailableExperienceProb = experienceSection.map(_._2.finalProb).sum
      val avgScore = allScores.sum / allScores.size
      val ranks: Seq[(TemplateMember, Double)] = experienceSection.map{case (t, pc) => (t, pc.scores.filter(_ > avgScore).sum)}.toSeq.sortBy(_._2)
      println(ranks.map{case (tm, score) => tm.name+ " " + score}.mkString(","))
      println(ranks.zipWithIndex.foldLeft((Nil, amountOfAvailableExperienceProb)){case (availableProb, ((tm, score), rank)) =>
        val currentProb = availableProb._2.value / 2
        (availableProb._1 ++ tm, currentProb)
      }.mkString(",")
      )
      //During reestimation of probabilities we need weighed function based on how many successfull experiments were with each template.( e.g. credit or trust)
      val totalNumberOfCredits = ???

    }
  }

}


case class ProbabilityContext(finalProb: Probability, scores: Seq[Double])
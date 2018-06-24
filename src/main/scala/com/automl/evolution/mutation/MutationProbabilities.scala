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

  var maxExperiencedProbabilityValue: Double = 0

  def update(updatedMember: TemplateMember, score: Double): Unit = {
    noveltySection.remove(updatedMember) match {
      case Some(noveltyPob) => //Case when there were no experience before

        //At first we just add new credit item. We move amount of probability from novelty to experience section. Then we will recalculate all probabilities.
        experienceSection += ((updatedMember, ProbabilityContext(noveltyPob, Seq(score))))
        val allScores = experienceSection.flatMap(_._2.scores) //TODO implement as incremental avg count?
        val amountOfAvailableExperienceProb = experienceSection.map(_._2.finalProb.value).sum

        val avgScore = allScores.sum / allScores.size
        if(allScores.forall(_ == avgScore)) {
          // Here we are ignoring the fact that some members could have more then one same score that should have given more credit(probability) to it
          val newEqualProb = amountOfAvailableExperienceProb / experienceSection.size
          maxExperiencedProbabilityValue = newEqualProb
          experienceSection.foreach { case (tm, probCtx) =>
            experienceSection.update(tm, ProbabilityContext(Probability(newEqualProb), probCtx.scores :+ score))
          }
        }
        else {
          //During reestimation of probabilities we need weighed function based on how many successfull experiments were with each template.( e.g. credit or trust)

          // 1) We are estimating how many runs were more successful than average and sum those `above average diffs` up.
          // Example: if we have template that was good twice than it is more probable that we will add it during mutation(but with other dimensions initialized to new values) again.
          // 2) Some templates don't have scores that are above avg and sum will be 0. It will cause NaN problem.
          // 3) Here we are under assumption that the higher score the better. Need to be carefull with different evaluation metrics.
          val templateToAboveAvgSums: Seq[(TemplateMember, Double)] =
            experienceSection
              .map{case (t, pc) => (t, pc.scores.filter(_ > avgScore)
                .map(v => v - avgScore).sum)}.toSeq.sortBy(-_._2)

          val templateToWeighedByRankSums = templateToAboveAvgSums.zipWithIndex.map{case ((tm, aboveAvgSum), rank) => (tm, aboveAvgSum / (rank +1))}

          val totalWeightedSum: Double = templateToWeighedByRankSums.map(_._2).sum

          //Normalizing and getting probabilities instead of weightedByRank sums.
          val templateToProbability = templateToWeighedByRankSums.map{ case (tm, weightedByRankSum) => (tm, (weightedByRankSum * amountOfAvailableExperienceProb) / totalWeightedSum)}

          //Update experienced probabilities and add score to history of current template
          templateToProbability.foreach{ case (tm, prob) =>
            val probContext = experienceSection(tm)
            experienceSection(tm) = ProbabilityContext(Probability(prob), probContext.scores )
          } //Maybe better to Keep two separate maps, One for probs and another for History of scores

          experienceSection.get(updatedMember).foreach{ probabilityContext =>
            experienceSection.update(updatedMember, ProbabilityContext(probabilityContext.finalProb, probabilityContext.scores :+ score))
          }

          maxExperiencedProbabilityValue = templateToProbability.map(_._2).max

          simpleNoveltySubsection.foreach{ case (tm, prob) => noveltySection.update(tm, Probability(maxExperiencedProbabilityValue))}
        }
      case None =>
        val newProbCtx = experienceSection.get(updatedMember).map( probCtx => ProbabilityContext(probCtx.finalProb, probCtx.scores :+ score))
          .getOrElse(throw new IllegalStateException("Template member should be either in novelty or experience section"))
        experienceSection.update(updatedMember, newProbCtx)
    }

    printTabular("NoveltySection", noveltySection)
    printTabular("ExperienceSection", experienceSection.map(v => (v._1, v._2.finalProb)))
    //Sanity check. Move it to the tests with scalacheck.
    require( simpleNoveltySubsection.forall(_._2.value >= maxExperiencedProbabilityValue), "Novelty within simple models should be always our first priority")
  }


  private def simpleNoveltySubsection = {
    noveltySection.filter(_._1.isInstanceOf[SimpleModelMember])
  }

  def rearrangeTemplatesByRank(ranks:  Seq[(TemplateMember, Double)], amountOfAvailableExperienceProb: Double) = {

  }


  def printTabular(title: String, values: mutable.Map[TemplateMember, Probability]) {
    // Print the list objects in tabular format.
    println(s"------------------------- $title -----------------------------------")
    printf("%-50s %-10s", "Template Member", "Probability")
    println()
    println("-----------------------------------------------------------------------------")
    values.foreach{ v =>
      println(f"${v._1.name}%-50s ${v._2.value}%-10f")
    }
    System.out.println("-----------------------------------------------------------------------------")
  }

}


case class ProbabilityContext(finalProb: Probability, scores: Seq[Double])
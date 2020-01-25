package com.automl.evolution.mutation

import com.automl
import com.automl.evolution.dimension.hparameter.{HPRangeWasExploredException, HyperParametersGroup, MutableHParameter}
import com.automl.evolution.diversity.DiversityStrategy
import com.automl.evolution.mutation.MutationKind.{AddLeafTemplateToNodeTemplate, MutateHPSOfLeafTemplate, MutateNodeTemplateToNodeTemplate, RemoveLeafTemplateFromNodeTemplate}
import com.automl.evolution.selection.{RankBasedSelectionProbabilityAssigner, RouletteWheel}
import com.automl.population.TPopulation
import com.automl.problemtype.ProblemType
import com.automl.template.ensemble.EnsemblingModelMember
import com.automl.template.simple.SimpleModelMember
import com.automl.template.{LeafTemplate, NodeTemplate, TemplateMember, TemplateTree}
import com.automl.{ConfigProvider, PaddedLogging}

import scala.util.Random


class RandomTemplateMutationStrategy(/*unused*/ diversityStrategy: DiversityStrategy, problemType: ProblemType, seed: Long)(implicit val logPaddingSize: Int) extends PaddedLogging {

  val tdConfig = ConfigProvider.config.getConfig("evolution.templateDimension")

  lazy val maxEnsembleDepth: Int = tdConfig.getInt("maxEnsembleDepth")
  lazy val pivotBetweenStructureAndHPMutations: Double = tdConfig.getDouble("pivotBetweenStructureAndHPMutations")
  val minNumberOfSubmembers = 2
  val maxNumberOfSubmembers = 10

  val hPMutationStrategy = new HPMutationStrategy()(4)

  val randomGenerator = new Random(seed)

  /**
    *
    * @param population Among other things it contains population-wide mutation probabilities
    * @return
    */
  def mutate(population: TPopulation, populationNotToIntersectWith: TPopulation = new TPopulation(Nil)): TPopulation = {

    info(s"Starting new mutation phase with RandomTemplateMutationStrategy for the population...")

    val mutatedIndividuals = population.individuals.foldLeft(List.empty[TemplateTree[TemplateMember]])((res, next) => {
      info(s"\nMutating \n${next.render} --------------------------------------------------------------------------\n")
      val newMutant = mutateIndividual(next)
      newMutant +: res
    })
    new TPopulation(mutatedIndividuals)
  }

  def getRandomBaseMemberBasedOnProblemType: SimpleModelMember = SimpleModelMember.randomMember(problemType)

  def getRandomBaseMemberWithExclusion(exclude: Seq[SimpleModelMember]): Option[TemplateMember] =
    SimpleModelMember.randomMemberWithExclusion(problemType, exclude)


  def mutateIndividual(individual: TemplateTree[TemplateMember]):TemplateTree[TemplateMember] = {

    def chooseLevelOfMutationUniformly( depth: Int): Int = {
      randomGenerator.nextInt(depth)
    }

    def mutateLeafToNode(lt: LeafTemplate[TemplateMember]) = {
      // We want at least 2 submembers and more
      val numberOfNewChildren = minNumberOfSubmembers + new Random().nextInt(3) // TODO Config
      val randomEnsemblingMember = EnsemblingModelMember.randomMember
      val oneForOriginalTemplate = 1
      info(s"\t\t Mutation happened from leaf node $lt to ensembling of ${numberOfNewChildren + oneForOriginalTemplate} submembers - $randomEnsemblingMember .")
      val nt = NodeTemplate(randomEnsemblingMember, Seq(lt) ++ (0 until numberOfNewChildren).map(_ => LeafTemplate(getRandomBaseMemberBasedOnProblemType)))
      //Setting new parent for all submembers including original leaf template
      nt.parent = lt.parent
      nt
    }

    def traverseAndMutate(individual: TemplateTree[TemplateMember], currentLevel: Int,
                          targetLevelOfMutation: Int): TemplateTree[TemplateMember] = individual match {
      case lt@LeafTemplate(member) =>
        if (targetLevelOfMutation == currentLevel) {

          val mutationTypeToPerform = if(currentLevel < maxEnsembleDepth - 1) randomGenerator.nextInt(3)  else randomGenerator.nextInt(2)// choosing uniformly one of the way to mutate

          //whether to choose uniformly or not it is a big question. Meta-learning?
          mutationTypeToPerform match {
            case 0 =>
              info(s"\t\t- Mutating hps of $lt at level = $currentLevel ")
              mutateHPGroupAspectOfTemplateTree(lt)
            case 1 =>
              info(s"\t\t- Mutating of $lt at level = $currentLevel with LeafToLeaf action( both structure and hps mutation")
              mutateLeafToLeaf(lt)
            case 2 =>
              info(s"\t\t- Mutating of $lt at level = $currentLevel resulted in a structure mutation, causing increasing of complexity")
              mutateLeafToNode(lt.asInstanceOf[LeafTemplate[SimpleModelMember]])
          }
        } else {
          lt // continue to next submember that will give us a way to a deeper level
        }

      case nt@NodeTemplate(ensemblingMember, subMembers) =>
        if (targetLevelOfMutation == currentLevel) {

          // Note: some mutation kinds are more likely than the others
          val validMutationKindsToPerform =
            if (subMembers.size == maxNumberOfSubmembers)
              Seq(RemoveLeafTemplateFromNodeTemplate, MutateHPSOfLeafTemplate, MutateHPSOfLeafTemplate)
            else if (subMembers.size == minNumberOfSubmembers) {
              Seq(AddLeafTemplateToNodeTemplate, MutateHPSOfLeafTemplate, MutateHPSOfLeafTemplate)
            }
            else {
              Seq(AddLeafTemplateToNodeTemplate, RemoveLeafTemplateFromNodeTemplate, MutateHPSOfLeafTemplate, MutateHPSOfLeafTemplate)
            }
          validMutationKindsToPerform.randElement match {
            case AddLeafTemplateToNodeTemplate => // add new submember
              info(s"\t\t- Mutate ensembling node at level = $currentLevel by adding new leaf template to its submembers")
              val additionalLT = LeafTemplate(getRandomBaseMemberBasedOnProblemType)
              val newNT = NodeTemplate(ensemblingMember, subMembers :+ additionalLT)
              newNT.parent = nt.parent
              newNT
            case RemoveLeafTemplateFromNodeTemplate => // add new submember
              info(s"\t\t- Mutate ensembling node at level = $currentLevel by removing random leaf template from its submembers")
              val newNT = NodeTemplate(ensemblingMember, subMembers.extractElem._2)
              newNT.parent = nt.parent
              newNT
            case MutateHPSOfLeafTemplate => // mutate Node's hyper parameters
              info(s"\t\t- Mutating hps of $nt (for NodeTemplate) at level = $currentLevel ") // TODO  TEST this case
              mutateHPGroupAspectOfTemplateTree(nt)

            case MutateNodeTemplateToNodeTemplate => nt //TODO mutate Node to Node. switch ensembling algo. We need it to cover mutations of 0 level ensemble node

          }
        } else { // Continue to random submember
          val randSubmember = subMembers.randElement
          val mutatedSubmember = traverseAndMutate(randSubmember, currentLevel + 1, targetLevelOfMutation)
          //TODO we are changing order here and it might change representation that we will be using as a key for caching
          val newNT = NodeTemplate(ensemblingMember, subMembers.diff(Seq(randSubmember)) :+ mutatedSubmember)

          newNT.parent = nt.parent
          newNT
        }

    }

    val targetLevelOfMutation = chooseLevelOfMutationUniformly(individual.height)

    val res = traverseAndMutate(individual, 0, targetLevelOfMutation)
    res
  }

  // TODO we should take into account complexity of template. If it is Leaf that is a stump, we probably want to mutate only its hyperparameters, otherwise we will just loose good template
  private def mutateLeafToLeaf(lt: LeafTemplate[TemplateMember]): LeafTemplate[TemplateMember] = {
    // TODO with exclusion we will have situation when we get read of good LeafTemplate
//    val randomBaseMemberBasedOnProblemType = getRandomBaseMemberWithExclusion(Seq(lt.member.asInstanceOf[SimpleModelMember])).asInstanceOf[Option[SimpleModelMember]]
    val randomBaseMemberBasedOnProblemType = Option(getRandomBaseMemberBasedOnProblemType)
    // TODO rewrite so that we don't need to cast member to SimpleModelMember
    randomBaseMemberBasedOnProblemType match {
      case Some(randomBaseMember) =>
        val newLeafTemplate = LeafTemplate(randomBaseMember)
        newLeafTemplate.parent = lt.parent
        info(s"\t\t Mutation happened from leaf node ${lt.render} to another leaf node ${newLeafTemplate.render}")
        //newLeafTemplate.internalHyperParamsMap //TODO ? should we copy something here or take some knowledge from others ??
        newLeafTemplate
      case None =>
        ???
//        if (currentLevel < maxEnsembleDepth - 1) {
//          info(s"\t\t Mutation happened from leaf node $lt to ensembling node due to randomBaseMemberBasedOnProblemType method returned NONE")
//          mutateLeafToNode(lt.asInstanceOf[LeafTemplate[SimpleModelMember]])
//        } else {
//          info("Dead end. Lost an opportunity to mutate. (dead end case 1)")
//          lt
//        }
      }
  }


  // TEmplateTree could be from any level of the ensemble tree. Which exactly level is decided outside of this method.
  def mutateHPGroupAspectOfTemplateTree(lt: TemplateTree[TemplateMember]): TemplateTree[TemplateMember] = {
    lt match {
      case template: NodeTemplate[_] =>
        info("!!!!!!! For now just do nothing as ensembling members do not have hps")
        template
      case template: LeafTemplate[_] =>
        //TODO make hpGroup non optional. If not provided we just need defualt values
        val member = lt.member
        val mutatedHPGroup: HyperParametersGroup[_ <: MutableHParameter[Double, _]] = member.hpGroupInternal.mutate()
        member.hpGroupInternal = mutatedHPGroup
        val leafTemplateWithMutatedHPS = LeafTemplate(member)
        info(s"\t\t- Mutating hps into $leafTemplateWithMutatedHPS")
        leafTemplateWithMutatedHPS
    }
  }
}


trait MutationKind

object MutationKind {
  case object AddLeafTemplateToNodeTemplate extends MutationKind
  case object RemoveLeafTemplateFromNodeTemplate extends MutationKind
  case object MutateHPSOfLeafTemplate extends MutationKind
  case object MutateNodeTemplateToNodeTemplate extends MutationKind
}

package com.automl.population

import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.problemtype.ProblemType
import com.automl.template.simple.SimpleModelMember
import com.automl.template.{LeafTemplate, TemplateMember, TemplateTree}

import scala.util.Random
import scala.reflect.runtime.{universe => ru}

case class PopulationBuilder[Item : ru.TypeTag, M : ru.TypeTag](population: M, problemType: Option[ProblemType] = None)(implicit ev: M <:< Population[Item]) {

  import scala.reflect.runtime.universe._
  val myTT = implicitly[TypeTag[Item]]
  val populationType = implicitly[TypeTag[M]]

  def withProblemType(problemType: ProblemType): PopulationBuilder[Item, M] = {
    this.copy(problemType = Some(problemType))
  }

  def withSize(populationSize: Int): PopulationBuilder[Item, M] = {
    val individuals = population.individuals
    val newIndividuals = if(populationSize >= population.individuals.size) {
      val rnd = new Random()
      val diff = populationSize - population.individuals.size
      val spanned = problemType.map(pt => Seq.fill(diff)(LeafTemplate(SimpleModelMember.randomMember(pt))))
        .getOrElse(Seq.fill(diff)(LeafTemplate(SimpleModelMember.randomMember())))
//      val spanned = Seq.fill(diff)(individuals(rnd.nextInt(individuals.size)))
      individuals ++ spanned

    } else {
      val rnd = new Random()
      val shrinked = Seq.fill(populationSize)(individuals(rnd.nextInt(individuals.size)))
      shrinked
    }

    if (populationType.tpe =:= typeOf[TPopulation]) {
      this.copy(population = new TPopulation(newIndividuals.asInstanceOf[Seq[TemplateTree[TemplateMember]]]).asInstanceOf[M])
    } else {
      this.copy(population = new HPPopulation(newIndividuals.asInstanceOf[Seq[HyperParametersField]]).asInstanceOf[M])
    }
  }

  // Consider using generalized type constraints like `def build(implicit ev: HasProperty =:= TTrue) = println(i)`.
  def build: M = population
}

object GenericPopulationBuilder {

  // Note without deferring type inference for M to the second list of parameters, type inference system will fail to understand
  // second type parameter If we put constraint `M <: Population[T]` right after(together)  T definition.
  // But it is still confusing... because it still does not infer T parameter from first parameter list.
  // So maybe deferred eimplicit constraint is for getting type T actually, even though it fills like we are just putting constrains on M
  def fromSeedPopulation[T : ru.TypeTag, M: ru.TypeTag](classifiersPopulation: M)(implicit ev: M <:< Population[T]): PopulationBuilder[T, M] = {
    PopulationBuilder[T, M](classifiersPopulation)
  }
}

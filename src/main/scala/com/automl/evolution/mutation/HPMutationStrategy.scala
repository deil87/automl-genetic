package com.automl.evolution.mutation

import com.automl.PaddedLogging
import com.automl.evolution.dimension.hparameter.HyperParametersField
import com.automl.population.HPPopulation

class HPMutationStrategy()(implicit val logPaddingSize: Int) extends PaddedLogging {

  //TODO notToIntersectWith is not used here
  // Use lenses here :)
  def mutate(population: HPPopulation, populationNotToIntersectWith: Option[HPPopulation] = None): HPPopulation = {
    new HPPopulation(
      population.individuals.map { hpField => {
        val newField = HyperParametersField(modelsHParameterGroups = hpField.modelsHParameterGroups.map { hpGroup =>
          hpGroup.mutate()
        })
        require(hpField.hashCode() != newField.hashCode(), "Hash codes should be different")
        debug(s"HyperParametersField mutated from $hpField to $newField")
        newField
      }
      })
  }
}

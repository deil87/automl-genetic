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

        // TODO maybe it is fine as other nodes in the tree could be different and repeating particular value within this node does not mean that we repead template as a whole
        debug("WARNING!!! We started to allow repeated values.")
//        require(hpField.hashCode() != newField.hashCode(), s"Hash codes for original and mutated hps should be different but were: $hpField vs $newField")
        debug(s"\t\t HyperParametersField mutated from $hpField to $newField")
        newField
      }
      })
  }
}

package com.automl.evolution.dimension

import com.automl.Population

trait EvolutionDimension {

  def evolve(population: Population)
  def applyMutation()
}





package com.automl.evolution.diversity

trait DistanceStrategy[DistanceObjectType, ReturnType] {
  def getDistance(objectOne: DistanceObjectType, objectTwo: DistanceObjectType): ReturnType
}

trait PhenotypicDistance[DistanceObjectType, ReturnType] extends DistanceStrategy[DistanceObjectType, ReturnType]

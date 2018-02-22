package com.automl.helper

sealed trait AppException

case class EvolutionTimeBoxTimeout() extends AppException

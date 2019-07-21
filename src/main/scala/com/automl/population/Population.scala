package com.automl.population

trait Population[+T] {
  def individuals: Seq[T]

  def size: Int = individuals.length

  def nonEmpty: Boolean = size != 0

  def render: Unit
}
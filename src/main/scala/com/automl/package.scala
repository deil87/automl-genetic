package com

import scala.util.Random

package object automl {

  implicit def iterableHelper[A](list: Seq[A]) = new {
    def randSubList = Random.shuffle(list).take(Random.nextInt(list.size) + 1)

    def randElement = if(list.nonEmpty) list(Random.nextInt(list.size)) else throw new IllegalStateException("randElement was called on the empty list")

    def extractElem = {
      val rand = randElement
      (rand, list.diff(Seq(rand)))
    }
  }

}

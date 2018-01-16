package com

import scala.util.Random

package object automl {

  implicit def iterableHelper[A](list: Seq[A]) = new {
    def randSubList = Random.shuffle(list).take(Random.nextInt(list.size) + 1)
    def randElement = list(Random.nextInt(list.size))
  }

}

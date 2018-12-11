package com.automl.helper


//NOTE maybe we don't nee this one. We can jump randomply to the children until we reach target depth level.
class TreeBreadthFirstFlattener {


  def flatten[T](s: Stream[T], f: T => Stream[T]): Stream[T] = {
    if (s.isEmpty) s
    else s.head #:: flatten(s.tail append f(s.head), f)
  }

}

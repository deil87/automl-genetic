package com.automl.regressor

class MajorityVoteRegressor extends EnsemblingRegressor[String]{

  //Note: it is important to have odd number of predictions
  override def evaluate(predictions: Seq[String]): String = {
    val map = predictions.groupBy[String](identity(_))
    map.toSeq.map{case (k, v) => (k, v.length)}.sortWith(_._2 > _._2).head._1
  }
}

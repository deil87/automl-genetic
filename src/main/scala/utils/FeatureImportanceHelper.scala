package utils

object FeatureImportanceHelper {

  def printFeatureScores(featureImportances: Array[(Int, Double)], indexToNameMapping: Array[String]) {


    val sortedFeatureScores: Seq[(Int, Double)] = featureImportances.sortBy(-_._2)

    val featureScoresInfo = sortedFeatureScores.map { case (featureIndex, score) =>
      val featureName = indexToNameMapping(featureIndex)

      (s"Score $score:", s"$featureName ($featureIndex)")
    }

    featureScoresInfo.foreach(line => printf("%-20s%s%n", line._1, line._2))
  }
}

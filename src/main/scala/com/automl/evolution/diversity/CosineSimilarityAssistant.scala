package com.automl.evolution.diversity

object CosineSimilarityAssistant {

  /*
  This version assumes we used a collect() method and it is not scalable solution. Consider transforming into IndexedRowMatrix and udf function.
 */
  def cosineSimilarity(vectorA: Seq[Double], vectorB: Seq[Double]): Double = {
    var dotProduct = 0.0
    var normA = 0.0
    var normB = 0.0
    var i = 0
    while ( {
      i < vectorA.length
    }) {
      dotProduct += vectorA(i) * vectorB(i)
      normA += Math.pow(vectorA(i), 2)
      normB += Math.pow(vectorB(i), 2)

      {
        i += 1; i - 1
      }
    }
    dotProduct / (Math.sqrt(normA) * Math.sqrt(normB))
  }
}

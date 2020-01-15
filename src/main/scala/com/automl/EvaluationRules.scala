package com.automl

import com.automl.problemtype.ProblemType
import com.automl.problemtype.ProblemType.{BinaryClassificationProblem, MultiClassClassificationProblem, RegressionProblem}

trait EvaluationRules {
  def theBiggerTheBetter(problemType: ProblemType): Boolean = {
    problemType match {
      case MultiClassClassificationProblem | BinaryClassificationProblem =>
        val config = ConfigProvider.config.getConfig("evolution")
        config.getString("evaluation.multiclass.metric") == "f1"
      case RegressionProblem => false
    }
  }
}

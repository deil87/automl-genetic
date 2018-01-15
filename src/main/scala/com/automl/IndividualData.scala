package com.automl

import com.automl.algorithm.{AlgorithmMember, AlgorithmTree}
import com.automl.helper.FitnessResult
import com.automl.template.{TemplateMember, TemplateTree}

case class IndividualData(template: TemplateTree[TemplateMember],
                          algorithm: AlgorithmTree[AlgorithmMember],
                          fitness: FitnessResult,
                          rank: Long,
                          probability: Double)

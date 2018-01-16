package com.automl

import com.automl.helper.FitnessResult
import com.automl.template.{TemplateMember, TemplateTree}

case class IndividualData(template: TemplateTree[TemplateMember],
                          algorithm: TemplateTree[TemplateMember],
                          fitness: FitnessResult,
                          rank: Long,
                          probability: Double)

package com.automl.population

import com.automl.template.{TemplateMember, TemplateTree}

class TPopulation(val individuals: Seq[ TemplateTree[TemplateMember]])
    extends Population[TemplateTree[TemplateMember]]{
}

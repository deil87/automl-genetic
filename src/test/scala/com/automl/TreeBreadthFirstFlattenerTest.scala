package com.automl

import com.automl.classifier.ensemble.bagging.SparkGenericBagging
import com.automl.helper.{TemplateTreeHelper, TreeBreadthFirstFlattener}
import com.automl.template._
import com.automl.template.ensemble.stacking.GenericStacking
import com.automl.template.simple.{Bayesian, GradientBoosting, LinearRegressionModel, RandomForest}
import org.scalatest.{FunSuite, Matchers}


class TreeBreadthFirstFlattenerTest extends FunSuite with Matchers{

  val individualTemplateTree =
    NodeTemplate(SparkGenericBagging(),
      Seq(
        LeafTemplate(new LinearRegressionModel()),
        LeafTemplate(new Bayesian()),
        NodeTemplate(new GenericStacking(),
          Seq(
            LeafTemplate(new GradientBoosting()),
            NodeTemplate(new GenericStacking(),
              Seq(
                LeafTemplate(new GradientBoosting())
              )
            )
          )
        ),
        LeafTemplate(new LinearRegressionModel()),
        LeafTemplate(new Bayesian())
      )
    )

  test("TreeBreadthFirstFlattener should flatten the tree") {
    def getMemberFun(individual: TemplateTree[TemplateMember]): Stream[TemplateTree[TemplateMember]] = individual match {
      case lt@LeafTemplate(_) => Stream(LeafTemplate(lt.member))

      case nt@NodeTemplate(ensemblingMember, subMembers) => nt.subMembers.toStream
    }
    val flattener = new TreeBreadthFirstFlattener().flatten[TemplateTree[TemplateMember]](Stream(individualTemplateTree), getMemberFun )

    val res = flattener.toList
    val result = res
  }



}


package com.weka

import org.scalatest.{FunSuite, Matchers}
import utils.test.TestBase
import weka.core.converters.ConverterUtils.DataSource
import weka.filters.Filter
import weka.filters.supervised.instance.SMOTE


class WEKASuite extends FunSuite with TestBase {

  ignore("Apply filter and fix imbalance in dataset") {

    val resource = System.getProperty("user.dir")+"/src/main/resources/material_training_weka.csv.arff"
    val observations = new DataSource(resource).getDataSet

    val classIndex: Int = observations.numAttributes() - 1
    observations.setClassIndex(classIndex)
    val smote = new SMOTE()

    smote.setRandomSeed(1234)
    smote.setPercentage(50)
    smote.setNearestNeighbors(5)
    smote.setClassValue("0")

    smote.setInputFormat(observations)

    val beforeSize = observations.size()
    val afterSMOTEdata = Filter.useFilter(observations, smote)

    beforeSize should be(afterSMOTEdata.size() * 2)
  }

}


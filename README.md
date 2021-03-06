# automl-genetic [![Build Status](https://travis-ci.org/deil87/automl-genetic.svg?branch=master)](https://travis-ci.org/deil87/automl-genetic)


AutoML is a pretty general term and could be thought of and implemented in many different ways with an exception of one thing being the same - they all should try to help us find best approaches(models) of solving a problem at hand.

Here we are trying to employ evolutionary algorithms and concepts to search the space of classifiers. In particularly we are interested in automatic construction of ensembles of classifiers because nowadays they have proved to be very efficient. 

Two of the most prominent technics to explore space of models are evolutionary approach and Sequential Model-Based Optimisation approach.

Couple of important notes about these two algos:
 - SMBO could be used only for relatively small hyperspaces as we need to materialise it in order to get predictions from a Surrogate model.
 - SMBO is unbiased as it considers all the entries in the grid on every iteration. Evolutionary approach is prone to stuck in a local min/max as it considers only entries that happened to be selected into the population.

Evolutionary approach (genetic) is the one that is being used in the current project. For those who are interested in the SMBO method and want to experiment with it there is an another project on my github account [smbo-java](https://github.com/deil87/smbo-java)

This project has it's roots growing from following [paper](https://link.springer.com/article/10.1007/s10994-017-5682-0) and basically represents an attempt to implement, experiment and extend those ideas and provide convenient framework with simple API for researchers and businesses.

Contributions of any kind are very welcome! Please contact me deil_87@mail.ru for coordination of our efforts.

<p align="center">
<img src="https://www.fit.cvut.cz/sites/default/images/favicon.png" width="60" height="60" "></p>
                                                                                            
 Supported by [FIT faculty of Czech Technical University](https://www.fit.cvut.cz/) in Prague.
 
 ## Demo
 
 <p align="center">
  <img src="https://github.com/deil87/automl-genetic/raw/master/doc/images/AutoMLMainTest_showcase.gif" width="100%"/>
</p>
 
 ## Getting started
 
 In order to run your own evolution process you can start with taking a look at example test com.automl.AutoMLSuite.
 
 In a nutshell you need to load and prepare you data with Spark and then pass it as a dataframe into core class com.automl.AutoML.
 
 
 Let's take a look at an example:
 ```
 val seed: Seq[LeafTemplate[SimpleModelMember]] = Seq(
         LeafTemplate(LinearRegressionModel()),
         LeafTemplate(Bayesian()),
         LeafTemplate(GradientBoosting()),
         LeafTemplate(DecisionTree())
       )

 val seedPopulation = new Population(seed)

 val autoMl = new AutoML(
        data = trainingSplit,
        maxTime = 200000,
        useMetaDB = false,
        initialPopulationSize = Some(7),
        seedPopulation = seedPopulation,
        maxGenerations = 5)
```

Here we are passing trainingSplit into AutoML, not using metaDB, set maximum search time to 200 seconds. Each evolution will perform 5 generations. Seed consist of 4 classifiers but initial size is 7. It means that there will be duplicates in our initial population.

Description of AutoML class input parameters:

| Parameter | Description |
| --- | --- |
| data | DataFrame with `features` and `label` colums. |
| maxTime | Time in milliseconds during which algorithm will be performing search of optimal ensemble. |
| useMetaDB | Whether or not to use metaDB of previously found templates based on similarity of datasets. |
| maxGenerations | Maximum number of generations (cycles of selection and mutation) within one evolution. After this number of generations algoritm will increase portion of `data` and essentially run new evolution.|
| seedPopulation | If we don't use metaDB ( `useMetaDB` is set to false) then we need to solve problem of cold start. Seed population is a set of classifiers that will be used to construct initial population. |
| initialPopulationSize | Size of initial population. Based on this value we will be drawing from the `seedPopulation` in order to get needed size. |


WebUI

In order to be able to run web interface AutoMLWeb class should be used. As of now there is no way to configure which dataset to use so this should be done manually. TODO
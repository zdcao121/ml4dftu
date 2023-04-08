# Machine Learning for Predicting U parameters #

## Introduction ##
This document describes the process of training machine learning model to predict the U parameters of a given dataset.

## Descriptor ##
The code to generate descriptors is available at: https://github.com/atomly-materials-research-lab/Descriptor. You can also use the descriptors that you define yourself.


## Machine Learning ##
We  use the random forest regression algorithm in the `scikit-learn` to predict the U parameters. If the size of the dataset is presumably enlarged by one or two 
orders of magnitude, the ML model could evolve into a deep neural network, meanwhile, the model accuracy, extrapolation, and generalization can be greatly enhanced.
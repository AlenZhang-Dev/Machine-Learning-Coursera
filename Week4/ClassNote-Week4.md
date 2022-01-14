# Week4

[TOC]

# Neural Networks: Representation

## Model Represent I

Neuron model: Logistic unit
Our hypothesis output is the logistic function applied to the sum of the values of our activation nodes, which have been multiplied by yet another parameter matrix Θ(2) containing the weights for our second layer of nodes.

- Each layer gets its own matrix of weight.

![Week4/Screen_Shot_2020-11-30_at_6.38.58_PM.png](Week4/Screen_Shot_2020-11-30_at_6.38.58_PM.png)

## Model Representation II

Forward propagation: Vectorized implementation

To re-iterate, the following is an example of a neural network:

![Week4/Screen_Shot_2020-11-30_at_6.51.19_PM.png](Week4/Screen_Shot_2020-11-30_at_6.51.19_PM.png)

![Week4/Screen_Shot_2020-11-30_at_6.52.34_PM.png](Week4/Screen_Shot_2020-11-30_at_6.52.34_PM.png)

![Week4/Screen_Shot_2020-11-30_at_6.52.56_PM.png](Week4/Screen_Shot_2020-11-30_at_6.52.56_PM.png)

![Week4/Screen_Shot_2020-11-30_at_6.42.49_PM.png](Week4/Screen_Shot_2020-11-30_at_6.42.49_PM.png)

## Examples and Intuitions

Use neural net to implement AND, NOR and OR.

![Week4/Screen_Shot_2020-11-30_at_6.54.44_PM.png](Week4/Screen_Shot_2020-11-30_at_6.54.44_PM.png)

Then we have the XNOR operator using a hidden layer with two nodes. The following summarizes the above algorithms:

![Week4/Screen_Shot_2020-11-30_at_6.55.36_PM.png](Week4/Screen_Shot_2020-11-30_at_6.55.36_PM.png)

## Multiclass Classification

To classify data into multiple classes, we let our hypothesis function return a vector of values. Say we wanted to classify our data into one of four categories. We will use the following example to see how this classification is done. This algorithm takes as input an image and classifies it accordingly:

![Week4/Screen_Shot_2020-11-30_at_6.58.05_PM.png](Week4/Screen_Shot_2020-11-30_at_6.58.05_PM.png)

![Week4/Screen_Shot_2020-11-30_at_6.59.20_PM.png](Week4/Screen_Shot_2020-11-30_at_6.59.20_PM.png)

## Quiz

1.

![Week4/Screen_Shot_2020-11-30_at_7.09.35_PM.png](Week4/Screen_Shot_2020-11-30_at_7.09.35_PM.png)

A and D

B: every input x has the chance to output 1.

C: at least 3 layer.

2.

![Week4/Screen_Shot_2020-11-30_at_7.11.29_PM.png](Week4/Screen_Shot_2020-11-30_at_7.11.29_PM.png)

A

3.

![Week4/Screen_Shot_2020-11-30_at_7.14.17_PM.png](Week4/Screen_Shot_2020-11-30_at_7.14.17_PM.png)

A

4.

![Week4/Screen_Shot_2020-11-30_at_7.15.31_PM.png](Week4/Screen_Shot_2020-11-30_at_7.15.31_PM.png)

![Week4/Screen_Shot_2020-11-30_at_7.15.39_PM.png](Week4/Screen_Shot_2020-11-30_at_7.15.39_PM.png)

A

5.

![Week4/Screen_Shot_2020-11-30_at_7.17.53_PM.png](Week4/Screen_Shot_2020-11-30_at_7.17.53_PM.png)

A.
# Week7 Support Vector Machines

# Overview

This week, we will be learning about support vector machine (SVM) algorithm. 

SVM are considered by many to be the most powerful 'black box' learning algorithm, and by posing a cleverly-chosen optimization objective, one of the most widely used learning algorithms today.

- Compare to both logistic regression and neural networks, the Support Vector Machine sometimes gives a cleaner, and sometimes more powerful way of learning algorithms.

# Optimization objective

Alternative view of logistic regression:

Hypothesis:  $h_{\theta}(x) = \frac{1}{1+e{-\theta^{T}x}}$

- if  $y = 1$, we want $h_{\theta} \approx 1, \theta^{T}x\gg0$
- if  $y = 0$, we want $h_{\theta} \approx 0, \theta^{T}x\ll0$

The cost function:

- $-(y\log h_{\theta}(x)+(1-y)log(1-h_{\theta}))$      ⇒ $-y\log \frac{1}{1+e{-\theta^{T}x}} - (1-y)\log \frac{1}{1+e{-\theta^{T}x}}$

The graph below:

![Week7/Screen_Shot_2021-01-14_at_10.32.31_PM.png](Week7/Screen_Shot_2021-01-14_at_10.32.31_PM.png)

Logistic regression:

$min_{\theta}
 \frac{1}{m}\left [   \sum_{i=1}^{m}-y\log \frac{1}{1+e{-\theta^{T}x}} - (1-y)\log \frac{1}{1+e{-\theta^{T}x}} \right ] + \frac{\lambda}{2m}\sum_{j=1}^{n}\theta_{j}^{2}$

Support vector machine hypothesis:

$min_{\theta}
C   \sum_{i=1}^{m}\left [-ycost_{1}(\theta^{T}x^{(i)}) - (1-y)cost_{0}(\theta^{T}x^{(i)}) \right ] + \frac{\lambda}{2m}\sum_{j=1}^{n}\theta_{j}^{2}$

# Large Margin Classification

![Week7/Screen_Shot_2021-01-14_at_10.58.21_PM.png](Week7/Screen_Shot_2021-01-14_at_10.58.21_PM.png)

![Week7/Screen_Shot_2021-01-14_at_10.42.49_PM.png](Week7/Screen_Shot_2021-01-14_at_10.42.49_PM.png)

Question:

![Week7/Screen_Shot_2021-01-13_at_10.25.44_AM.png](Week7/Screen_Shot_2021-01-13_at_10.25.44_AM.png)

# The mathematics behind large margin classification

![Week7/Screen_Shot_2021-01-14_at_10.44.36_PM.png](Week7/Screen_Shot_2021-01-14_at_10.44.36_PM.png)

Notes: Same as **vector inner product**

# Kernels

![Week7/Screen_Shot_2021-01-14_at_10.46.00_PM.png](Week7/Screen_Shot_2021-01-14_at_10.46.00_PM.png)

It's **Gaussian Kernel**

![Week7/Screen_Shot_2021-01-14_at_10.48.35_PM.png](Week7/Screen_Shot_2021-01-14_at_10.48.35_PM.png)

![Week7/Screen_Shot_2021-01-14_at_10.49.15_PM.png](Week7/Screen_Shot_2021-01-14_at_10.49.15_PM.png)

**SO HOW DO WE CHOOSE THE LANDMARK?**

![Week7/Screen_Shot_2021-01-14_at_10.50.28_PM.png](Week7/Screen_Shot_2021-01-14_at_10.50.28_PM.png)

SVM with Kernels

![Week7/Screen_Shot_2021-01-14_at_10.53.32_PM.png](Week7/Screen_Shot_2021-01-14_at_10.53.32_PM.png)

![Week7/Screen_Shot_2021-01-14_at_10.53.40_PM.png](Week7/Screen_Shot_2021-01-14_at_10.53.40_PM.png)

Parameters in SVM

![Week7/Screen_Shot_2021-01-14_at_10.54.17_PM.png](Week7/Screen_Shot_2021-01-14_at_10.54.17_PM.png)

Logistic regression or SVMs?

![Week7/Screen_Shot_2021-01-14_at_10.54.49_PM.png](Week7/Screen_Shot_2021-01-14_at_10.54.49_PM.png)

Quiz: Support Vector Machines

![Week7/Screen_Shot_2021-01-14_at_5.07.14_PM.png](Week7/Screen_Shot_2021-01-14_at_5.07.14_PM.png)

![Week7/Screen_Shot_2021-01-14_at_5.07.26_PM.png](Week7/Screen_Shot_2021-01-14_at_5.07.26_PM.png)

![Week7/Screen_Shot_2021-01-14_at_5.07.38_PM.png](Week7/Screen_Shot_2021-01-14_at_5.07.38_PM.png)

![Week7/Screen_Shot_2021-01-14_at_5.07.50_PM.png](Week7/Screen_Shot_2021-01-14_at_5.07.50_PM.png)

![Week7/Screen_Shot_2021-01-14_at_5.07.58_PM.png](Week7/Screen_Shot_2021-01-14_at_5.07.58_PM.png)
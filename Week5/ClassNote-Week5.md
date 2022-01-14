# Week5

[TOC]

# Cost Function

![Week5/Screen_Shot_2020-12-04_at_9.02.26_AM.png](Week5/Screen_Shot_2020-12-04_at_9.02.26_AM.png)

![Week5/Screen_Shot_2020-12-03_at_10.58.53_PM.png](Week5/Screen_Shot_2020-12-03_at_10.58.53_PM.png)

# Backpropagation Algorithms

"Backpropagation" is neural-network terminology for minimizing our cost function, just like what we were doing with gradient descent in logistic and linear regression. 

Our goal is to compute:

![Week5/Screen_Shot_2020-12-04_at_9.07.09_AM.png](Week5/Screen_Shot_2020-12-04_at_9.07.09_AM.png)

![Week5/Screen_Shot_2020-12-04_at_9.07.23_AM.png](Week5/Screen_Shot_2020-12-04_at_9.07.23_AM.png)

We use following algorithms to achieve our goals.

![Week5/Screen_Shot_2020-12-04_at_9.08.06_AM.png](Week5/Screen_Shot_2020-12-04_at_9.08.06_AM.png)

## Intuition

![Week5/Screen_Shot_2020-12-04_at_3.25.01_PM.png](Week5/Screen_Shot_2020-12-04_at_3.25.01_PM.png)

![Week5/Screen_Shot_2020-12-04_at_3.23.37_PM.png](Week5/Screen_Shot_2020-12-04_at_3.23.37_PM.png)

# Implementation

## Unrolling

![Week5/Screen_Shot_2020-12-04_at_4.41.59_PM.png](Week5/Screen_Shot_2020-12-04_at_4.41.59_PM.png)

## Gradient Checking

Assure out backpropagation works as intended.

![Week5/Screen_Shot_2020-12-04_at_4.43.04_PM.png](Week5/Screen_Shot_2020-12-04_at_4.43.04_PM.png)

![Week5/Screen_Shot_2020-12-04_at_4.44.37_PM.png](Week5/Screen_Shot_2020-12-04_at_4.44.37_PM.png)

![Week5/Screen_Shot_2020-12-04_at_4.45.05_PM.png](Week5/Screen_Shot_2020-12-04_at_4.45.05_PM.png)

```matlab
epsilon = 1e-4;
for i = 1:n,
  thetaPlus = theta;
  thetaPlus(i) += epsilon;
  thetaMinus = theta;
  thetaMinus(i) -= epsilon;
  gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
end;
```

## Random Initialization

Initializing all theta weights to zero does not work with neural networks. When we backpropagate, all nodes will update to the same value repeatedly. Instead we can randomly initialize our weights for our $Θ$ matrices using the following method.

![Week5/Screen_Shot_2020-12-05_at_11.01.10_AM.png](Week5/Screen_Shot_2020-12-05_at_11.01.10_AM.png)

![Week5/Screen_Shot_2020-12-05_at_11.05.00_AM.png](Week5/Screen_Shot_2020-12-05_at_11.05.00_AM.png)

Implementation Note:

- Implement backprop to compute DVec
- Implement numerical gradient check to compute gradApprox.
- Make sure they give similar values.
- turn off gradient checking. Using backprop code for learning.

## Put all together

![Week5/Screen_Shot_2020-12-05_at_11.18.44_AM.png](Week5/Screen_Shot_2020-12-05_at_11.18.44_AM.png)

When we perform forward and back propagation, we loop on every training example:

```matlab
for i = 1:m,
   Perform forward propagation and backpropagation using example (x(i),y(i))
   (Get activations a(l) and delta terms d(l) for l = 2,...,L
```

![Week5/Screen_Shot_2020-12-05_at_11.19.30_AM.png](Week5/Screen_Shot_2020-12-05_at_11.19.30_AM.png)

# Quiz

1.

![Week5/Screen_Shot_2020-12-05_at_11.20.15_AM.png](Week5/Screen_Shot_2020-12-05_at_11.20.15_AM.png)

根据反向传播的向量公式即可。

![Week5/Screen_Shot_2020-12-05_at_11.24.56_AM.png](Week5/Screen_Shot_2020-12-05_at_11.24.56_AM.png)

2.

![Week5/Screen_Shot_2020-12-05_at_11.25.49_AM.png](Week5/Screen_Shot_2020-12-05_at_11.25.49_AM.png)

3.

![Week5/Screen_Shot_2020-12-05_at_11.27.20_AM.png](Week5/Screen_Shot_2020-12-05_at_11.27.20_AM.png)

![Week5/Screen_Shot_2020-12-05_at_11.45.29_AM.png](Week5/Screen_Shot_2020-12-05_at_11.45.29_AM.png)

4.

![Week5/Screen_Shot_2020-12-05_at_11.51.56_AM.png](Week5/Screen_Shot_2020-12-05_at_11.51.56_AM.png)

5. ![Week5/Screen_Shot_2020-12-05_at_12.05.11_PM.png](Week5/Screen_Shot_2020-12-05_at_12.05.11_PM.png)

Choose B and C.

Reference：

[神经网络15分钟入门！--反向传播到底是怎么传播的？](https://zhuanlan.zhihu.com/p/66534632)
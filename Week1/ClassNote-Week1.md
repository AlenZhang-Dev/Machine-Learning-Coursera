# Week 1

[TOC]

# Introduction

Introduce the core idea of teaching a computer to learn concepts using data - without being explicitly programmed.

**linear regression with one variable.** Discuss the application of linear regression to housing price prediction, present the notion of a cost function, also introduce the gradient descent method for learning.

The course require the knowledge of l**inear algebra** concepts.

---

## What's Machine Learning?

Andrew NG: Machine Learning is the science of getting computers to learn, without being explicitly programmed.

- Grew out of work in AI
- New capability for computers

Examples:

- Database mining
  
  - Large datasets from growth of automation/web
  - E.g., Web click data, medical records, biology, engineering

- Applications can't program by hand.
  
  - E.g., Autonomous helicopter, handwriting recognition, most of Natural Language Processing(NLP), Computer Vision.

- Self-customizing programs
  
    E.g., Amazon, Netflix product recommendations.

- Understanding human learning(brain, real AI).

**What's Machine Learning?[Definition]**

Two definitions of Machine Learning are offered.

- Arthur Samuel described it as: **"the field of study that gives computers the ability to learn without being explicitly programmed."** This is an older, informal definition.
- Tom Mitchell provides a more modern definition: **"A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."**

Examples:

playing checkers.

E = the experience of playing many games of checkers

T = the task of playing checkers.

P = the probability that the program will win the next game.

Suppose your email program watches which emails you do or do not mark as spam, and based on that learns how to better filter spam. Explain the E T P according to Tom Mitchell's definiation.

E = Watching you label emails as spam or not spam.

T = Classify emails as spam or not spam.

P = The number(or fraction) of emails correctly classified as spam/not spam.

In general, any machine learning problem can be assigned to one of two broad classifications:

**Supervised learning and Unsupervised learning.**

Machine Learning algorithms 

- Supervised Learning
- Unsupervised Learning

Others: Reinforcement Learning, recommender systems.

## Supervised Learning

In supervised learning, we are given a data set and already know what our correct output should look like, **having the idea that there is a relationship between the input and the output.**

<u>Supervised learning problems are categorized into "**regression**" and "**classification**" problems.</u> 

In a regression problem, we are trying to predict results within a **continuous output**, meaning that we are trying to map input variables to some continuous function. 

In a classification problem, we are instead trying to predict results in a **discrete output[离散输出]**. In other words, we are trying to map input variables into discrete categories.

![Week1/Screen_Shot_2020-10-28_at_3.50.40_PM.png](Week1/Screen_Shot_2020-10-28_at_3.50.40_PM.png)

Regression exp

![Week1/Screen_Shot_2020-10-28_at_3.51.03_PM.png](Week1/Screen_Shot_2020-10-28_at_3.51.03_PM.png)

Classification

**Example 1:**

Given data about the size of houses on the real estate market, try to predict their price. Price as a function of size is a continuous output, so this is a regression problem.

We could turn this example into a classification problem by instead making our output about whether the house "sells for more or less than the asking price." Here we are classifying the houses based on price into two discrete categories.

**Example 2**:

(a) Regression - Given a picture of a person, we have to predict their age on the basis of the given picture

(b) Classification - Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.

Practice：

You’re running a company, and you want to develop learning algorithms to address each of two problems. 

Problem 1: You have a large inventory of identical items. You want to predict how many of these items will sell over the next 3 months.

Problem 2: You'd like software to examine individual customer accounts, and for each account decide if it has been hacked/compromised. Should you treat these as classification or as regression problems?

Answer: 

Treat problem 1 as a regression problem, problem 2 as a classification problem.

Explain:

First one : we have thousand of items [Continuous]

Second one: 0 - not hacked, 1 - hacked.[Hacked or not, Discrete]

## Unsupervised Learning

Unsupervised learning is a type of machine learning that looks for previously undetected patterns in a data set with no pre-existing labels and with a minimum of human supervision.    —- Wikipedia

Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

We can derive this structure by clustering the data based on relationships among the variables in the data.

With unsupervised learning there is no feedback based on the prediction results.

**Example:**

Clustering: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.

Non-clustering: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music from a mesh of sounds at a [cocktail party](https://en.wikipedia.org/wiki/Cocktail_party_effect)).

The Following examples using an unsupervised learning algorithm: 

Given a set of news articles found on web, group them into sets of articles about the same stories.

Given a database of customer data, automatically discover market segments and group customers into different market segments.

# Model and Cost Function

## Model Representation

**Notes:**

![Week1/Screen_Shot_2020-10-27_at_6.03.23_PM.png](Week1/Screen_Shot_2020-10-27_at_6.03.23_PM.png)

Linear regression model with only one variable[Also called univariate linear regression]

![Week1/Screen_Shot_2020-10-27_at_6.10.45_PM.png](Week1/Screen_Shot_2020-10-27_at_6.10.45_PM.png)

**Material:**

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function h : X → Y so that h(x) is a “good” predictor for the corresponding value of y. For historical reasons, this function h is called a **hypothesis**. Seen pictorially, the process is therefore like this:

When the target variable that we’re trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem. When y can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.

## Cost Function

**Notes:**

How to choose the parameters?

![Week1/Screen_Shot_2020-10-27_at_6.17.14_PM.png](Week1/Screen_Shot_2020-10-27_at_6.17.14_PM.png)

**Material**:

We can measure the accuracy of our hypothesis function by using a **cost function**. This takes an average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's and the actual output y's.

![Week1/Screen_Shot_2020-10-28_at_4.06.02_PM.png](Week1/Screen_Shot_2020-10-28_at_4.06.02_PM.png)

MSE: the most common one used for regression problems.

[Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)

## Cost Function Intuition I

1.Set the theta(0) to zero.

![Week1/Screen_Shot_2020-10-27_at_6.58.03_PM.png](Week1/Screen_Shot_2020-10-27_at_6.58.03_PM.png)

2.The optimization objective for our learning algorithm is we want to choose the value of theta (1).  The Minimizes J(theat(1)).

![Week1/Screen_Shot_2020-10-27_at_7.39.12_PM.png](Week1/Screen_Shot_2020-10-27_at_7.39.12_PM.png)

Material:

![Week1/Screen_Shot_2020-10-27_at_7.39.36_PM.png](Week1/Screen_Shot_2020-10-27_at_7.39.36_PM.png)

![Week1/Screen_Shot_2020-10-27_at_7.39.52_PM.png](Week1/Screen_Shot_2020-10-27_at_7.39.52_PM.png)

## Cost function - Intuition II

1. Keep both two parameters of the cost function.

![Week1/Screen_Shot_2020-10-28_at_4.42.52_PM.png](Week1/Screen_Shot_2020-10-28_at_4.42.52_PM.png)

The cost function may be look like below:

![Week1/Screen_Shot_2020-10-28_at_4.43.16_PM.png](Week1/Screen_Shot_2020-10-28_at_4.43.16_PM.png)

Contour plots / figure

A contour plot is a graph that contains many contour lines. A contour line of a two variable function has constant value at all points of same line. An example of such a graph is the one to the right below:

![Week1/Screen_Shot_2020-10-28_at_4.46.54_PM.png](Week1/Screen_Shot_2020-10-28_at_4.46.54_PM.png)

3 green x have the same cost.

When θ0=360 and θ1 = 0, the value of J(θ0, θ1) in the contour plot gets closer to the center thus reducing the cost function error. Now Giving our hypothesis function a slightly positive slope result in a better fit of the data.

![Week1/Screen_Shot_2020-10-28_at_4.57.37_PM.png](Week1/Screen_Shot_2020-10-28_at_4.57.37_PM.png)

The graph above minimizes the cost function as much as possible and consequently, the result of θ1 and θ0 tend to be around 0.12 and 250 respectively. Plotting those values on our graph to the right seems to put our point in the center of the inner most 'circle'.

# Parameter Learning

## Gradient Descent

The algorithm called gradient descent for **minimizing the cost function j().** 

Gradient is a general algorithm.

![Week1/Screen_Shot_2020-10-27_at_8.07.15_PM.png](Week1/Screen_Shot_2020-10-27_at_8.07.15_PM.png)

The outline of Gradient descent algorithm:

Understand the concept between Assignment and Truth assertion.

Simultaneous update both parameters.

![Week1/Screen_Shot_2020-10-27_at_8.29.22_PM.png](Week1/Screen_Shot_2020-10-27_at_8.29.22_PM.png)

**Material:**

![Week1/Screen_Shot_2020-10-27_at_8.31.07_PM.png](Week1/Screen_Shot_2020-10-27_at_8.31.07_PM.png)

We will know that we have succeed when our cost function is at the very bottom of the pits in our graph. when its values is the minimum. The red arrows show the minimum points in the graph.

The way we do is by taking the derivative of our cost function. The slope of the tangent is the derivative at the point and it will give us a direction move towards. We make steps down the cost function in the direction with the steepest descent. The size of each step is determined by the parameter α, which is called the learning rate.

![Week1/Screen_Shot_2020-10-27_at_8.31.16_PM.png](Week1/Screen_Shot_2020-10-27_at_8.31.16_PM.png)

## Gradient Descent Intuition

Apply gradient descent to minimize our squared cost function.

Gradient descent can converge to local minimum, even with the learning rate a fixed.

don't need to change the a.

Explain the Gradient descent with one variable:

![Week1/Screen_Shot_2020-10-27_at_9.02.06_PM.png](Week1/Screen_Shot_2020-10-27_at_9.02.06_PM.png)

![Week1/Screen_Shot_2020-10-27_at_9.01.29_PM.png](Week1/Screen_Shot_2020-10-27_at_9.01.29_PM.png)

Features of Gradient descent:

When reach the bottom of our convex function, the derivative will always be 0, thus we get the settled *θ1.*

![Week1/Screen_Shot_2020-10-27_at_9.01.42_PM.png](Week1/Screen_Shot_2020-10-27_at_9.01.42_PM.png)

## Gradient Descent For Linear Regression

Simplify the function:

When we apply the Gradient descent algorithm with Linear Regression Model, we will derived a new form of the gradient descent equation.

![Week1/Screen_Shot_2020-10-28_at_6.37.54_PM.png](Week1/Screen_Shot_2020-10-28_at_6.37.54_PM.png)

![Week1/Screen_Shot_2020-10-27_at_9.10.52_PM.png](Week1/Screen_Shot_2020-10-27_at_9.10.52_PM.png)

![Week1/Screen_Shot_2020-10-27_at_9.16.44_PM.png](Week1/Screen_Shot_2020-10-27_at_9.16.44_PM.png)

Convex function equals to Bowl shaped

"Batch" Gradient Descent.

![Week1/Screen_Shot_2020-10-27_at_9.24.15_PM.png](Week1/Screen_Shot_2020-10-27_at_9.24.15_PM.png)

Material:

A new form of the gradient descent equation:

![Week1/Screen_Shot_2020-10-28_at_6.41.58_PM.png](Week1/Screen_Shot_2020-10-28_at_6.41.58_PM.png)

A single example of derivative.

![Week1/Screen_Shot_2020-10-28_at_6.42.55_PM.png](Week1/Screen_Shot_2020-10-28_at_6.42.55_PM.png)

So this is simply gradient descent on the original cost function j. This method looks at every example in the entire training set on every step, which is called **batch gradient descent.【批量梯度下降】** Note that, while gradient descent can be susceptible to **local minima** in general. the optimization problem we have posed here for linear regression has only on global, and no other local, optima; thus gradient descent always converges to the global minimum. 

Indeed, j is a **convex quadratic function**. Here is an example of gradient descent as it run to minimize a quadratic function.

![Week1/Untitled.png](Week1/Untitled.png)

The ellipses shown above are the contours of a quadratic function. Also shown is the trajectory taken by gradient descent, which was initialized at (48,30). The x’s in the figure (joined by straight lines) mark the successive values of θ that gradient descent went through as it converged to its minimum.

Rating Test:

![Week1/Screen_Shot_2020-10-28_at_8.31.33_PM.png](Week1/Screen_Shot_2020-10-28_at_8.31.33_PM.png)

![Week1/Screen_Shot_2020-10-28_at_8.35.23_PM.png](Week1/Screen_Shot_2020-10-28_at_8.35.23_PM.png)

![Week1/Screen_Shot_2020-10-28_at_8.37.06_PM.png](Week1/Screen_Shot_2020-10-28_at_8.37.06_PM.png)

![Week1/Screen_Shot_2020-10-28_at_9.01.07_PM.png](Week1/Screen_Shot_2020-10-28_at_9.01.07_PM.png)

Choose C and D

![Week1/Screen_Shot_2020-10-28_at_8.53.44_PM.png](Week1/Screen_Shot_2020-10-28_at_8.53.44_PM.png)

![Week1/Screen_Shot_2020-10-28_at_9.09.25_PM.png](Week1/Screen_Shot_2020-10-28_at_9.09.25_PM.png)

[Answers](https://www.notion.so/1cd7ea007e554bc59178ca74776018d5)

# Linear Algebra Review

## Matrices and Vectors

Matrix: Rectangular array of numbers.

Matrices are 2-dimensional arrays.

![Week1/Screen_Shot_2020-10-29_at_8.28.31_AM.png](Week1/Screen_Shot_2020-10-29_at_8.28.31_AM.png)

Vector: An n x 1 matrix.

A vector is a matrix with one column and many rows.

So vectors are a **subset** of matrices.

![Week1/Screen_Shot_2020-10-29_at_8.34.06_AM.png](Week1/Screen_Shot_2020-10-29_at_8.34.06_AM.png)

**Notation and terms[Essential]**

![Week1/Screen_Shot_2020-10-29_at_8.40.18_AM.png](Week1/Screen_Shot_2020-10-29_at_8.40.18_AM.png)

**Matlab commands below:**

```matlab
% The ; denotes we are going back to a new row.
A = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12]

% Initialize a vector 
v = [1;2;3] 

% Get the dimension of the matrix A where m = rows and n = columns
[m,n] = size(A)

% You could also store it this way
dim_A = size(A)

% Get the dimension of the vector v 
dim_v = size(v)

% Now let's index into the 2nd row 3rd column of matrix A
A_23 = A(2,3)
```

## Addition and Scalar Multiplication

**Outline:**

Matrix addition and subtraction, and how to multiply a matrix by a number. SO called Scalar Multiplication.

**Matrix Addition:**

![Week1/Screen_Shot_2020-10-29_at_8.49.42_AM.png](Week1/Screen_Shot_2020-10-29_at_8.49.42_AM.png)

**Scalar Multiplication:**

![Week1/Screen_Shot_2020-10-29_at_8.51.49_AM.png](Week1/Screen_Shot_2020-10-29_at_8.51.49_AM.png)

**Combination of Operands:**

![Week1/Screen_Shot_2020-10-29_at_8.55.22_AM.png](Week1/Screen_Shot_2020-10-29_at_8.55.22_AM.png)

**Matlab command:**

```matlab
% Initialize matrix A and B 
A = [1, 2, 4; 5, 3, 2]
B = [1, 3, 4; 1, 1, 1]

% Initialize constant s 
s = 2

% See how element-wise addition works
add_AB = A + B 

% See how element-wise subtraction works
sub_AB = A - B

% See how scalar multiplication works
mult_As = A * s

% Divide A by s
div_As = A / s

% What happens if we have a Matrix + scalar?
add_As = A + s

A =
   1   2   4
   5   3   2

B =
   1   3   4
   1   1   1
s =  2

add_AB =
   2   5   8
   6   4   3

sub_AB =
   0  -1   0
   4   2   1

mult_As =
    2    4    8
   10    6    4

div_As =
   0.50000   1.00000   2.00000
   2.50000   1.50000   1.00000

add_As =
   3   4   6
   7   5   4
```

## Matrix Vector Multiplication

The number of columns of matrix must match the number of rows of vector.

![Week1/Screen_Shot_2020-10-29_at_9.11.24_AM.png](Week1/Screen_Shot_2020-10-29_at_9.11.24_AM.png)

m x n matrix multiply n x 1 matrix [n-dimensional vector] = m-dimensional vector

## Matrix Matrix Multiplication

**Details:**

![Week1/Screen_Shot_2020-10-29_at_9.26.48_AM.png](Week1/Screen_Shot_2020-10-29_at_9.26.48_AM.png)

**Apply:**

![Week1/Screen_Shot_2020-10-29_at_9.37.41_AM.png](Week1/Screen_Shot_2020-10-29_at_9.37.41_AM.png)

**Matlab command:**

```matlab
% Initialize a 3 by 2 matrix 
A = [1, 2; 3, 4;5, 6]

% Initialize a 2 by 1 matrix 
B = [1; 2] 

% We expect a resulting matrix of (3 by 2)*(2 by 1) = (3 by 1) 
mult_AB = A*B

% Make sure you understand why we got that result
A =
     1     2
     3     4
     5     6

B =
     1
     2

mult_AB =
     5
    11
    17
```

## Matrix Multiplication Properties

**Not commutative**

![Week1/Screen_Shot_2020-10-29_at_9.42.59_AM.png](Week1/Screen_Shot_2020-10-29_at_9.42.59_AM.png)

**Associative**

(A * B) * C = A * (B * C)

![Week1/Screen_Shot_2020-10-29_at_9.46.25_AM.png](Week1/Screen_Shot_2020-10-29_at_9.46.25_AM.png)

**Identity Matrix:**

The identity matrix of size n is the n x n square matrix **with ones on the main diagonal and zeros elsewhere.
Details of Identity Matrix:**

![Week1/Screen_Shot_2020-10-29_at_9.55.37_AM.png](Week1/Screen_Shot_2020-10-29_at_9.55.37_AM.png)

## Inverse and Transpose

Start by how it relates to real numbers.

**Inverse Matrix:**

![Week1/Screen_Shot_2020-10-29_at_10.07.41_AM.png](Week1/Screen_Shot_2020-10-29_at_10.07.41_AM.png)

- Matrices that don't have an inverse are **"singular" or "degenerate"**

So how can we get the inverse matrix?

Sometimes you can compute inverses by hand but almost no one does that those day. It turns out there is very good **numerical software** for taking a matrix and computing.

**Matrix Transpose:**

The transpose of a matrix is an operator which flips a matrix over its diagonal.

![Week1/Screen_Shot_2020-10-29_at_10.14.30_AM.png](Week1/Screen_Shot_2020-10-29_at_10.14.30_AM.png)

Matlab Command

```matlab
% Initialize matrix A 
A = [1,2,0;0,5,6;7,0,9]

% Transpose A 
A_trans = A' 

% Take the inverse of A 
A_inv = inv(A)

% What is A^(-1)*A? 
A_invA = inv(A)*A

A =
     1     2     0
     0     5     6
     7     0     9
A_trans =

     1     0     7
     2     5     0
     0     6     9
A_inv =

    0.3488   -0.1395    0.0930
    0.3256    0.0698   -0.0465
   -0.2713    0.1085    0.0388
A_invA =

    1.0000   -0.0000    0.0000
    0.0000    1.0000   -0.0000
   -0.0000         0    1.0000
```

---

# FAQ for Week1:

**Q1: In the cost function, why don't we use absolute value(or mod(), or some other function) instead of the squared error?**

The absolute value has some bad characteristics for  minimization.

- The gradient is not continuous because the absolute value function is not differentiable at its minimum point.
- It does not emphasize the correction of large errors.
- The abs() function is also not very mathematically efficient.

However, the squared error cost function has some very desirable characteristics:

- The cost can be computed very efficiently.
- Its partial derivative is easily computed.
- Its partial derivative is continuous.

**Q2: How does the Cocktail Party problem work?**

It isn't clustering.

Clustering would not preserve the time-sequence of the sound samples, so the results would not sound like speech.

The method used is Principal Component Analysis. This method is a mathematical trick that takes two sets of correlated data, and returns two new sets of data that are not correlated. The sequence of the data is retained.

If the data is audio recordings, the result has the effect to the human ear of sounding like two separate audio tracks.
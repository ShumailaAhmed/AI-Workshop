# AI-Workshop
 Artificial Intelligence Workshop With Pytorch

## Introduction to tensors


## Notes 
Pytorch provides trnsor computation with strong gpu acceleration
It enables convinent implementation of nural networks

Tensor is a data structure, neural nets are fundamentally structuren as tensors.
A tensor is Generalization of matrices with n dimentions. 

![Basic Tensor Structure](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/tensors.png)


1. a zero dimentional vector is called scalar  
2. a one dimentional tensor is called vector   
3. a two dimentional tensors are matrices  
4. a 3d tensors called tensor.  

## Tensor Operations

Open google colab  

```!pip install torch
import torch

oneDTensor = torch.tensor([1,2,3])

print(oneDTensor)
print(oneDTensor.dtype)

```
 Indexing Tensors is similar to indexing python list
 
 ```
 print(oneDTensor[0])
 ```
Slicing the tensor is just like slicing the python list

```
oneDTensor = torch.tensor([1,2,3,4,5,6])
print(oneDTensor[:4])
```

we can also define float tensors

```
 floatTensor= torch.FloatTensor([1,2,3,4,5,6])
 print (floatTensor)
 print(floatTensor.dtype)
 ```
 
 size of tensor is determined by size method
 
 ```print(floatTensor.size())```
 
 
 reshape the tensor just like numpy.reshape function 
 
 ```floatTensor.view(6,1)
 floatTensor.view(6,2) #throws error
 floatTensor.view(3,2)
 floatTensor.view(2,-1)
 ```
 
 ### Convert numpy array into Tensors
 
 ```import numpy as np
 array = np.array([1,2,3,4,5])
 tensor= torch.from_numpy(array) #convert numpy array to tensor
 print(tensor)
 numpy = tensor.numpy()
 print(numpy)
 ```
 
 ### Vector Operations
 
 ```
 t1 = torch.tensor([1,2,3])
t2 = torch.tensor([1,2,3])
 ```
 
 these one dimentional tensors behave like vectors, such that if we add these vectors, each homologous value is added, similar is the case with tensor multiplication and scalar multiplication.
 
 ```t1+t2
 t1 * t2
 t1*10
 
 ```
 the dot product is given as
 
 ```
 dot = torch.dot(t1,t2)
 print(dot)
 ```
 torch linspace prints 100 equally spaced numbers between specified ranges.

 ```
  torch.linspace(0,1000)
 ```
 
 we can also explicitly spicify spacing by third parameter
 
 ```
  torch.linspace(0,1000,5)
 ```
 
 we can plot the ranges using matplotlib
  
```import matplotlib.pyplot as plt
x=  torch.linspace(0,10)
y= torch.exp(x)
plt.plot(x.numpy(),y.numpy())
#sin plot
y= torch.sin(x)
plt.plot(x.numpy(),y.numpy())


```

### Two Dimentional Tensors

2D tensors are analogous to matrices, having some number of rows and some number of columns. Gray scale images are typical example of 2D tensors.these contain values from 0 to 255 in a single channel of information, hence these can be stored in the 2 dimentional tensors or matrices.

![2d Tensor example Gray-scale Image](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/grayscale.gif)

Tensors can be extended to 3, 4 and n-dimentions.  

Lets initialize a 1D tensor

```one_d = torch.arange(2,7)
one_d
```
we can also specify the step size

```one_d = torch.arange(2,7,2)
one_d
```

we can arrange 1d tensor as 2d tensor

```
oneDim = torch.arange(0,9)
twoDim = oneDim.view(3,3)
```

we can check the dimentionality of a tensor by using the dim method

```
twoDim.dim()
```

indexing the 2D tensor can be accomplished by

```
twoDim[0,2]
```
Next we can define a 3d array as follows
```
x= torch.arange(18).view(2,3,3)
x
```
this reshapes 2 blocks each having 3 rows and 3 column, If we want to reshape it into 3 blocks having 2 rows and 3 columns we can accomplish this by 
```
x= torch.arange(18).view(3,2,3)
x
```
similarly we can have 2 blocks of 3 rows and 2 columns by 
```
x= torch.arange(18).view(3,3,2)
x
```

### Slicing Multidimentional Tensors

we can select single element from a 3D tensor as follows
```
x[1,1,1]
```
if we want to slice a multidimentional tesnor we ca follow suit

```
x[1,0:2,0:3]
#or
 x[1,:,:]
```

### Matrix Multiplication

we can perform matrix multiplication between two matrices A and B if and only if the number of columns in A is equal to number of rows in matrix B

the resulting matrix will have rows_A x col_B size

```
matA = torch.tensor([0,3,4,5,5,2]).view(2,3)
matA

matB = torch.tensor([3,4,3,-2,4,-2]).view(3,2)
matB

torch.matmul(matA,matB)
#or
matA @ matB
```

### Derivatives (Gradients)

The derivatives represent the functions rate of change. The derivative at a point x is defined as slope of the tangent to the curve at x. 

This can be achieved via python code

```
x = torch.tensor(2.0, requires_grad= True)
y =  9*x**4 + 2*x**3 + 3*x**2 + 6*x +1
y.backward()
x.grad
```
Now if we consider partial derivatives, 

```
x = torch.tensor(1.0, requires_grad= True)
z = torch.tensor(2.0, requires_grad= True)

y = x**2 + z**3

y.backward()
print(x.grad)
print(z.grad)
```

This is all, we can now use this knowlegde to train neural networks. 

## Linear Regression

We will get fimiliar with common machine learning algorithms and train a linear model to propoerly fit a set of data points. Here we will discuss various fundamental concepts involved in training a model, including

1. loss function  
2. gradient descent
3. optimization
4. learning rate, etc

### Basic Concepts

#### What is Machine learning? 

It is the concept of building computational algorithms that can learn overtime based on expirence. Such that rather than explicitly programming a hardcoded set of instructions, an intelligent system is given the capacity learn, detect and predict meaningful patterns. 

#### Supervised Learning
Supervised learning makes use of datasets with labelled features which define meaning of the training data. Hence when introduced to new data, the algorithm is able to produce a corresponding output. 

![Supervised Learning](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/supervisedlearning.gif)


### Make Predictions using defined model

we know the equation of a line can be modeled using the equation  
**y = wx+b** some of you might have seen it in the form **y = mx+c**  
Here w is the slope and b is the y intercept of the line, these parameters actually define the line. So in linear regression, our ultimate goal is to predict these parameters for a given set of datapoint

In other words, we use the data points to train the linear model to have optimal weight and bias value to provide us with the line of best fit. 

```
import torch

w = torch.tensor(3.0,requires_grad=True)
b = torch.tensor(1.0,requires_grad=True)

def forward(x):
  y = w*x+b
  return y
  
x= torch.tensor(2)
forward(x)

x= torch.tensor([[2],[4],[7]])
forward(x)

```

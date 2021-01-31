# AI-Workshop
 Artificial Intelligence Workshop With Pytorch

## Introduction to tensors


## Notes 
Pytorch provides trnsor computation with strong gpu acceleration
It enables convinent implementation of nural networks

Tensor is a data structure, neural nets are fundamentally structuren as tensors.
A tensor is Generalization of matrices with n dimentions. 

![Basic Tensor Structure](https://github.com/ShumailaAhmed/AI-Workshop/main/tensors.png?raw=true)


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

![2d Tensor example Gray-scale Image](https://raw.githubusercontent.com/ShumailaAhmed/AI-Workshop/main/grayscale.gif?token=ALS5FOAAD6FXSTLMUI4YIDDAC2WPE)

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

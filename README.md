# AI-Workshop
 Artificial Intelligence Workshop With Pytorch

## Introduction to tensors


## Notes 
Pytorch provides tensor computation with strong GPU acceleration
It enables the convenient implementation of neural networks

Tensor is a data structure, neural nets fundamentally are tensors.
A tensor is the Generalization of matrices with n dimensions. 

![Basic Tensor Structure](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/tensors.png)


1. a zero-dimensional vector is called scalar  
2. a single-dimensional tensor is called a vector   
3. two-dimensional tensors are matrices  
4. a 3d tensors called a tensor.  

## Tensor Operations

Open google colab  

```!pip install torch
import torchstructuren

oneDTensor = torch.tensor([1,2,3])

print(oneDTensor)
print(oneDTensor.dtype)

```
 Indexing Tensors is similar to indexing a python list
 
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
 
 size of a tensor is determined by the size method
 
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
 
 these one dimensional tensors behave like vectors, such that if we add these vectors, each homologous value is added, similar is the case with tensor multiplication and scalar multiplication.
 
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
 
 we can also explicitly specify spacing by the third parameter
 
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

### Two Dimensional Tensors

2D tensors are analogous to matrices, having some number of rows and some number of columns. Grayscale images are the typical example of 2D tensors. These contain values from 0 to 255 in a single channel of information, hence these can be stored in the 2-dimensional tensors or matrices.

![2d Tensor example Gray-scale Image](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/grayscale.gif)

Tensors can be extended to 3, 4, and n-dimensions.  

Let's initialize a 1D tensor

```one_d = torch.arange(2,7)
one_d
```
we can also specify the step size

```one_d = torch.arange(2,7,2)
one_d
```

we can arrange a 1d tensor as a 2d tensor

```
oneDim = torch.arange(0,9)
twoDim = oneDim.view(3,3)
```

we can check the dimensionality of a tensor by using the dim method

```
twoDim.dim()
```

indexing the 2D tensor can be accomplished by

```
twoDim[0,2]
```
Next, we can define a 3d array as follows
```
x= torch.arange(18).view(2,3,3)
x
```
this reshapes 2 blocks each having 3 rows and 3 columns, If we want to reshape it into 3 blocks having 2 rows and 3 columns we can accomplish this by 
```
x= torch.arange(18).view(3,2,3)
x
```
similarly, we can have 2 blocks of 3 rows and 2 columns by 
```
x= torch.arange(18).view(3,3,2)
x
```

### Slicing Multidimensional Tensors

we can select a single element from a 3D tensor as follows
```
x[1,1,1]
```
if we want to slice a multidimensional tensor we can follow suit

```
x[1,0:2,0:3]
#or
 x[1,:,:]
```

### Matrix Multiplication

we can perform matrix multiplication between two matrices A and B if and only if the number of columns in A is equal to the number of rows in matrix B

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

The derivatives represent the function rate of change. The derivative at a point x is defined as the slope of the tangent to the curve at x. 

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

This is all, we can now use this knowledge to train neural networks. 

## Linear Regression

We will get familiar with common machine learning algorithms and train a linear model to properly fit a set of data points. Here we will discuss various fundamental concepts involved in training a model, including

1. loss function  
2. gradient descent
3. optimization
4. learning rate, etc

### Basic Concepts

#### What is Machine learning? 

It is the concept of building computational algorithms that can learn over time based on experience. Such that rather than explicitly programming a hardcoded set of instructions, an intelligent system is given the capacity to learn, detect and predict meaningful patterns. 

#### Supervised Learning
Supervised learning makes use of datasets with labeled features that define the meaning of the training data. Hence when introduced to new data, the algorithm can produce a corresponding output. 

![Supervised Learning](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/supervisedlearning.gif)


### Make Predictions using a defined model

we know the equation of a line can be modeled using the equation  
**y = wx+b** some of you might have seen it in the form **y = mx+c**  
Here w is the slope and b is the y-intercept of the line, these parameters define the line. So in linear regression, our ultimate goal is to predict these parameters for a given set of the datapoint

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

### Standard Way to Define a linear model 
The standard way is to use the linear model class

```import torch
from torch.nn import Linear
```

```
torch.manual_seed(1)# set seed for reproducibility
model = Linear(in_features=1, out_features=1) #for every prediction we make, for everyoutput there is a single input
print(model.bias, model.weight)
```

```x = torch.tensor([2.0])
print(model(x))
```
```
# making multiple predictions at once
x = torch.tensor([[2.0],[3.3],[4.0]])
print(model(x))
```

### Writing Custom Modules
```
import torch
import torch.nn as nn
#########################

class LR(nn.Module): #the class LR will inherit from Module, LR will be sub class of nn.module and will inherit methods and variables from parent class
#initialize the class 
  def __init__(self, input_size, output_size): ## define input and output classes 
    super().__init__() ## while using inheritence, it is customary to initialize the parent class
    self.linear = nn.Linear(input_size, output_size)
  def forward(self, x): #to make predictions we make use of forward method
    pred = self.linear(x)
    return pred
    
 ########################
 
#see the model structure 
torch.manual_seed(1)
model = LR(1, 1)
print(list(model.parameters()))

#########################
#make predictions
x =torch.tensor([1.0])
print(model.forward(x))

#multiple predictions
x =torch.tensor([[1.0],[2.0]])
print(model.forward(x))
```

So we have seen 3 ways to use a simple linear regression model or determine the equation of a line. but this is fundamental to define any complex model that we will be doing shortly.  

Next, we perform training to learn new weights and bias values based on given data. Up until this point we have not introduced any data and we were just looking at random parameters.   

### Making Data 
Before we learn the concepts related to the model and train a model on the data we need to create and visualize the data, so let's do that
we are going to make the following data  
![Data Visualization](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/data.png)


```
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
###################

#each data point has a x and a y value
#we will be specifying x value by using randn
X = torch.randn(100, 1)*10 # returns a tensor filled with random numbers that are normally distributed
#it will have 100 rows and 1 column, they are y default centered around zero, to change that we multiply
#it by 10 to scale the values
y = X  #this creates an x = y line with normally distributed values, but this is not very challenging 
#y = X + 3*torch.randn(100, 1) #this adds noise in the data by shifting y value either upward or downward, so that the noise is alo normally distributed 
#and since randn centers around zero, with a small standard deviation, so to make noise reasonable significant we multiply by 3
plt.plot(X.numpy(), y.numpy(), 'o')
plt.ylabel('y')
plt.xlabel('x')
#Now that we have created the dataset, we need to train a model to fit this dataset. Before going to that let's see how our previous definition of the model fits the data 
###################
#re use the class we created earlier
class LR(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.linear = nn.Linear(input_size, output_size)
  def forward(self, x):
    pred = self.linear(x)
    return pred
###################
    #we have random weights and bias assigned to model 
  torch.manual_seed(1)
model = LR(1, 1)
print(list(model.parameters()))
###################
we can see what value is actually assigned to the parameters
[w, b] = model.parameters()
def get_params():
  return (w[0][0].item(), b[0].item()) this will return the two values as tuple
# we are doing this because these are the parameters we will update or try to change in order to FIT the data.
###################
we draw the line defined by our random parameters and see how well it fits the data
def plot_fit(title):
  plt.title = title
  w1, b1 = get_params()
  x1 = np.array([-30, 30])
  y1 = w1*x1 + b1
  plt.plot(x1, y1, 'r')
  plt.scatter(X, y)
  plt.show()
###################

plot_fit('Initial Model') 
```
As we can see the current model is a very bad fit for the data so we need to use GRADIENT DESCENT to update its parameters


### Loss Function

Let's Define our Goal, provided a set of data points we need to find the parameters of a line that will fit this data adequately. 
Since our random initialization does not fit the data we need some sort of OPTIMIZATION algorithm that will adjust the parameters based on the total error.
We will continue this until we get a line with the least error. 

For every data point, the error is given by the difference between predicted and true value. 

It is obtained by subtracting the prediction at a point by the actual Y value, the greater the difference, the greater is the error. 

![Error in prediction](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/error.png)

The prediction is given as the 
```math
y' = w*X+b
```

The error between the predicted and actual value is given as 

```math
Loss = (y-y')^2
Loss = (y-w*X+b)^2
```
Consider a point (3,-3), according to our weight and bias value, the predicted value 

```math
y' = 0.5152631998062134*3-0.44137823581695557
y' = 1.1044113636 
```

The loss can be calculated as 

```math
Loss = (3-1.1044113636)^2
Loss = 3.59325627845
```
### Gradient Descent

Our goal is to minimize this loss to ideally zero, or as close to zero as possible. Next, we need to find a way to train our model to determine weight parameters that will minimize the error function the most.  

The answer is the GRADIENT DESCENT!  

Here are the steps
1. Initialize the Linear model with random weights 
2. Based on the error associated with these initial parameters we want to move in the direction that gives us the smallest error.

![Gradient Descent](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/gradientdescent.gif)

If we take the gradient of our error function (i.e the derivative), the slope of the tangent at the current value takes us in the direction of the highest error so we move in the direction -ve to the gradient, that will take us in the direction of the lowest error. So we take the current weight and subtract the derivative of that function at that point.   

 We descent with the gradient, but to ensure optimal results we should descend in very small steps. For this, we multiply the gradient with a very small number called as **Learning Rate** 
 
 ### Learning Rate
 
 The standard starting values of the learning rate is 1/10. The learning rate ensures that we are converging adequately, a high learning rate might result in divergent behavior. We adjust the learning rate based on empirical results. 
 
  
### Mean Squared Error
![Mean Squared Error](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/mse.gif)

The mean squared error is the summation of the error for each point, given by the following equation.  

![Error Equation](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/mean_squared_error.svg)


substituting the prediction equation we get the following loss as a function of the parameters w and b.
![Error Equation](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/lossfunction.jpeg)

In mean squared error, we take the average over the data points (n)  
  
### Updating the parameters

For every iteration, the new weight  w1 is given by 
  
```math
m1 = m0 - LR * f'(w)
```
while the new bias term is given by 

```math
m1 = b0 - LR * f'(b)
```
We don't have to worry about the math when we code, its just to know what is going behind the scene. 


### Code Implementation
First, we define the loss function and optimizer to reduce the loss
```
### continued from last section
criterion = nn.MSELoss() #we will access the builtin loss function Mean Squared Loss
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) 
# we use an optimizer to update the parameter
#we use Stochastic gradient descent algorithm 
#Stochastic gradient descent vs batch gradient descent 
#batch gradient descent finds error for everysingle data point which is computationally inefficient
#sgd minimizes gradient descent one sample at a time, it will update weights more often.

# in sgd we pass two arguments, one the model parameters we want to optimize and the learning rate
```
Next, we train the model

```
#epochs are the no of single pass through entire data set
epochs = 100 #we will train the model for 100 epochs
#as we itterate through the dataset we calculate the error function and backpropagate the gradient of of error function
#1 epoch = underfits
#if we do too many epochs we get overfitting
losses = []
for i in range(epochs): 
  y_pred = model.forward(X) # get pred
  loss = criterion(y_pred, y) #get mse 
  print("epoch:", i, "loss:", loss.item()) #print epoch and loss
  
  losses.append(loss) #keep track of losses, append in a list to visualize decrease in loss
  optimizer.zero_grad() #we must call zero grad because gradients accumulate following the loss.backward call
  loss.backward() # compute the derivative
  optimizer.step() # update the parameters, all optimizers implement the step

```

To visualize the loss wrt to the number of epochs, we can plot it

```
plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch')
```

To visualize how good our output model fits the data points we can call the plot_fit function we made in the last section 
```
plot_fit("Trained Model")
```

If this is your first time applying a machine learning algorithm it is a pretty big step, Congratulations!  

Right now we trained a model to fit a set of data points. The next step is to learn how to use a Linear model to classify between two discrete classes of data points.

## Introduction To Perceptron 

A perceptron is a single-layered neural network, the most basic form of a neural network. NNs are inspired by biological neural networks. 

### What is deep learning 

Deep learning leverages the use of deep neural networks. It emulates the way we humans learn, it is widely used to learn patterns from observational data.  
The deeper the network, the more complex information the network can learn. 


![Gradient Descent](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/neuralnet.png)

### Creating Dataset
To import the dataset we will use sklearn. Sklearn provides access to many preprepared datasets.
```
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn import datasets
```
For manipulating the dataset we will use numpy and for plotting and visualization we will make use of matplotLib

```
  n_pts = 100 #define noumber of points we want
  centers = [[-0.5, 0.5], [0.5, -0.5]]#define the center cord for cluster, we have two clusters so we need two centeroids, nested list
  X, y = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centers, cluster_std=0.4) #creates a cluster of data points randomly centered around a defined centerpoint
  #random state is similar to seed for reproducibility
  #std is the standard deviation of points from respective centerpoint
  #the data lies in X and labels lies in y
  print(X)
  print(y) #y contains the labels 0 and 1 
  x_data = torch.Tensor(X)
  y_data = torch.Tensor(y.reshape(100, 1))
  #print(y.shape)
```
To visualize the data we can plot the data we have just created

```
def scatter_plot():
  plt.scatter(X[y==0, 0], X[y==0, 1]) #plot the data points with label 0, do a boolean check and g
  plt.scatter(X[y==1, 0], X[y==1, 1]) #plot the 
  
scatter_plot()
```
![Data for classification](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/dataForClassification.png)


Now that we have the data we need to classify, let's take a look at how we are going to accomplish the task and what are the underlying concepts involved.  

The task of classification uses previously labeled data to learn how to classify new unseen data. A model like linear regression starts with some random initialization and makes a prediction that is most likely wrong. Then the model will be trained through some optimization function, through many iterations until it reaches the parameter values which perform considerably well. We use previously labeled data to train the model, and get predictions on new data that does not have a label. Model is the equation of a line, we predict on basis of whether the point lies above or below the line. Refer to the following image

![Classification](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/classification.gif)


The randomly initiated line calculates the error and based on the magnitude of error, it re-adjusts the parameters of the model to minimize the error, but **how do we calculate the error? **  

We use a continuous error function and assign the miss-classified points with a big penalty. The bigger the error, the bigger the penalty, hence the error variations will account for the direction we need to move into, to reduce the error function.  

The total error is the _sum of penalties associated with each point_ . 

### Concept of Cross-Entropy Loss
Mathematically we calculate the loss using cross-entropy function. 
1. For each point, if it lies above the line it has a probability closer to 0  
2. For each point if the point lies below the line it has a probability closer to 1
3. Each point is assigned some probability 
4. To compute cross-entropy loss we find the summation of the logarithm of each probability 

![Cross Entropy Loss Equation](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/crossentropyloss.png)

The Perceptron model can be shown as follows 

![Perceptron](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/percepton.png)

The input features and parameters are passed into the neuron to predict the output, which is then passed through the activation function. 

### Implementation 

```
#continued from previous section 
class Model(nn.Module): #define a model class as we did earlier
    def __init__(self, input_size, output_size): #add arguments input size and output size as with perceptron structure 
      super().__init__() #for class inheritence
      self.linear = nn.Linear(input_size, output_size)
    def forward(self, x): #To make prediction we define forward function
      pred = torch.sigmoid(self.linear(x)) #for each forward pass we pass the x through linear function and then through sigmoid activation function to get the probabilities
      return pred #return the prediction
      
torch.manual_seed(2) #for reproducibility
model = Model(2, 1) # pass 2 input features and get one pred output
print(list(model.parameters())) # we get weight 1 weight 2 and bias
#these are not optimal

[w, b] = model.parameters() #extract the parameters by unpacking into list of two elements
w1, w2 = w.view(2) # unpack tuple into w1 and w2
def get_params():
  return (w1.item(), w2.item(), b[0].item()) #get the python number from tensor values
```
to visualize we can use the following code making utility of matplotlib

```
def plot_fit(title):
  plt.title = title
  w1, w2, b1 = get_params()
  x1 = np.array([-2, 2])
  x2 = (w1*x1 + b1)/(-w2) # equation of the line 
  plt.plot(x1, x2, 'r')
  scatter_plot()
  plt.show()
  
plot_fit('Initial Model')  
```

As we can see the initial model is not the best fit for our data, so we will apply gradient descent to train it.

![Initial Parameters before optimization](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/initial_model.png)


### Model Training 

As we discussed the criterion by which we will compute the error of the model is a cross-entropy loss. Since we have two classes, we will use binary cross-entropy loss. 

```
criterion = nn.BCELoss() #binary loss cross entropy  
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # define loss function Stochastic gradient descent
```
Time to train the model, we will train for a specified no of epochs, we iterate through data set, calculate error function, backpropagate the gradient of error and update weight and bias.

```
epochs = 1000
losses = []

for i in range(epochs):
  y_pred = model.forward(x_data) #grab prediction
  loss = criterion(y_pred, y_data) #get loss as per criteria binary cross entropy
  print("epoch:", i, "loss:", loss.item()) #print epoch and loss
  losses.append(loss.item()) 
  optimizer.zero_grad() # avoid accumulation
  loss.backward() #compute the gradient
  optimizer.step() # update params, the step method is used to do so
```
Notice the raining code is very similar to the one we did in linear regression, this is because the process does not change while training the neural model. Either we classify the data into discrete classes or fit a model into a continuous set of data points it does not change the training process majorly.  

Plot the training process to visualize, 

```
plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.grid()
```
![Training Process](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/training progress.png)

Let's see how well we fit our data. 

```
plot_fit("Trained Model")
```
![Initial Parameters before optimization](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/final_model.png)

As we can see we now fit the model much adequately. 


### Model Testing

We might also want to choose new unlabelled data and let the model predict its class. Do so we can do as follows

```
point1 = torch.Tensor([1.0, -1.0])
point2 = torch.Tensor([-1.0, 1.0])
plt.plot(point1.numpy()[0], point1.numpy()[1], 'ro')
plt.plot(point2.numpy()[0], point2.numpy()[1], 'ko')
plot_fit("Trained Model")
#blue are class 0
#red point class 1
print("Red point positive probability = {}".format(model.forward(point1).item())) 
print("Black point positive probability = {}".format(model.forward(point2).item())) 


```
As we can see our model gives confirming probabilities, we also male predict method in our model class

```
class Model(nn.Module): #define a model class as we did earlier
    def __init__(self, input_size, output_size): #add arguments input size and output size as with perceptron structure 
      super().__init__() #for class inheritence
      self.linear = nn.Linear(input_size, output_size)
    def forward(self, x): #To make prediction we define forward function
      pred = torch.sigmoid(self.linear(x)) #for each forward pass we pass the x through linear function and then through sigmoid activation function to get the probabilities
      return pred #return the prediction
    def predict(self, x):
      pred = self.forward(x) # and call the forward pass
      if pred >= 0.5: 
        return 1
      else:
        return 0        
```

and get the class prediction

```
print("Red point belongs in class {}".format(model.predict(point1))) 
print("Black point belongs in class = {}".format(model.predict(point2))) 
```

From here we train a much deeper neural network. Good Job on your very first NN implementation.


### Linearly inseparable data and Deep Neural Networks

Real data is not always linearly separable by a straight line, we at many instances will need non-linear boundaries as we get into more complex data. For this task, we will use deep neural networks. 

![Linearly inseparable data](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/nonlineardata.png)

The data can not be distinguished into two classes using a linear function hence we need to obtain a curve for the classification. To do so we need to combine two perceptrons

![Combining two models](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/linear_combination.gif)

linearly combining existing models to create new models that better classify our data is the core concept of a complex neural network. This means we will superimpose two linear models to form a simple nonlinear model.

### Core concepts
1. Forward pass
2. Activation function (in our case Sigmoid)
3. Error calculation (in our case Cross entropy loss)
4. Backpropagation
5. Optimization (in our case Stochastic)

### Architecture of a neural network
1. Input layer
2. Hidden Layers
3. Output Layers

### Common Activation Functions
1. Relu
2. leaky Relu
3. Tanh
4. Sigmoid 
etc

More the no of hidden layers an nn has, the deeper it gets. We can combine many layers to obtain a very complex model. A deep NN is nothing but a multilayered perceptron. The number of hidden layers is called the depth of the neural network. 

We can visualize the concepts on http://playground.tensorflow.org/ this will give us some intutive understanding of how the neural networks work. 

### Implementation

Like always start by importing relevant libraries, we are using the old code base with very few changes

```
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn import datasets
```
We will make the dataset using sklearn make circles function

```
n_pts = 500 #change the points to 500 from 100
X, y = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2) #we will use make circles instead of make blob
# third argument is noise refering to standard deviation of gaussian noise. 
# larger gaussian noise will cause data points to deviate from shape
# factor value means dia of inner will be 20 % of outer circle
x_data = torch.Tensor(X)
y_data = torch.Tensor(y.reshape(500, 1)) #we have 500 point
```

next we use previous code block to plot the data for visualization purpose 

```
# plotting data
def scatter_plot():
  plt.scatter(X[y==0, 0], X[y==0, 1])
  plt.scatter(X[y==1, 0], X[y==1, 1])
```
```
scatter_plot()
```
# add image

As we can see evidently the data set can not be seprated by a single straight line and we will need a non linear function to perform this task for us. to accomplish the task we need a deep neural network. In our perceptron model we had an input layer and output layer but for our nn, as we saw a while ago, we need some quantity of hidden layers. and pass our input through entire model to accomplish the forward pass.

```
class Model(nn.Module):
  def __init__(self, input_size, H1, output_size): # here we define H1 for a hidden layer 1, and pass the number of neurons in the first hidden layer 
    super().__init__()
    self.linear = nn.Linear(input_size, H1) #as we are working on fully connected network, the input layer is not connected to output layer it is connected to the hidden layer
    self.linear2 = nn.Linear(H1, output_size) # the hidden layer is then connected to the output layer. 
  def forward(self, x): # we have to account for input in forward function
    x = torch.sigmoid(self.linear(x)) # x is going to go through first layer and return prediction
    x = torch.sigmoid(self.linear2(x))#  that is passed through second layer to get final prediction
    # this is because we pass our inout through entire model.
    return x # finally we return the prediction
  def predict(self, x): 
    pred = self.forward(x)
    if pred >= 0.5:
      return 1
    else: 
      return 0
```
Next we initialize the model by passing in the number of nodes, and initiating the model with random parameters.

```
torch.manual_seed(2)
model = Model(2, 4, 1) # we pass 2 input nodes, 4 nodes in hidden layer and output will be 1 node
print(list(model.parameters())) # we can see we have many more weight and bias parameters
```

For this task we will replace our SDG optimizer with Adam optimizer. In SGD we have to very mindful about the learning rate. A very small learning rate leads to slow convergence and a very large learning might induce divergent behaviour. On the other hand Adam optimizer algorithm does not maintain single weight for all weight updates, it compute adaptive learning rate. It is very popular with large model and datasets. Its is a default for practice implementations, in research and academia. 

```
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # adam is a combination of two extensions of SDG adagrad and rmsprop , it is an adaptive algorithm
# use lr 0.1 
```

The training process remains same 

```
epochs = 1000
losses = []
for i in range(epochs):
  y_pred = model.forward(x_data)
  loss = criterion(y_pred, y_data)
  print("epoch:", i, "loss", loss.item())
  losses.append(loss.item())
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
```
plot the training process

```
plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch')
```

now we need to plot the output, we will need to plot the non linear boundries in form of contours. 

```
def plot_decision_boundary(X, y): # accepts two arguments
  x_span = np.linspace(min(X[:, 0]) -0.25, max(X[:, 0])+0.25) # define span of x of data min and max horizontal cord value 
  #+0.25 is tolerance to better visualize
  y_span = np.linspace(min(X[:, 1]) -0.25, max(X[:, 1])+0.25) # span of y min and max y coordinate
  xx, yy = np.meshgrid(x_span, y_span) #get square 2d array
  # take 50 x 1 and give 50  x 50 matrix
  # take 1d vector and return 2d vector, making repeated copies of rows
  grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()]) # flaten the xx and yy  and concat column wise and convert into tensor
  # arrange in id element, flatten them
  pred_func = model.forward(grid), feed entire grid into model for prediction
  z = pred_func.view(xx.shape).detach().numpy() # reshape to link to grid cord counterpart so that we can view as contour
  # detech to exclude any extra pred graph
  #now every single cord in graph will have a corredponding pred
  plt.contourf(xx, yy, z) # finally # plot distinct contour zone
```
plot the decision boundry along side scatter plot of original data points
 
```
plot_decision_boundary(X, y)
scatter_plot()
```
# add image

The white region is the -ve region, and a point lying in this region will have a value of 0, the point in black region will have a prediction 1
Now to finally test the model on an unlabelled unseen data 

```
x = 0.025 # horizontal cord of point
y = 0.025 # Y cord
point = torch.Tensor([x, y])# define point
prediction = model.predict(point) #get prediction
plt.plot([x], [y], marker='o', markersize=10, color="red")#  plot the point , not the scatter
print("Prediction is", prediction) # print prediction
plot_decision_boundary(X, y) # plot decesion boundry
```


# AI-Workshop
 Artificial Intelligence Workshop With Pytorch
https://colab.research.google.com

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
    
``` 
``` 


class LR(nn.Module): #the class LR will inherit from Module, LR will be sub class of nn.module and will inherit methods and variables from parent class
#initialize the class 
  def __init__(self, input_size, output_size): ## define input and output classes 
    super().__init__() ## while using inheritence, it is customary to initialize the parent class
    self.linear = nn.Linear(input_size, output_size)
  def forward(self, x): #to make predictions we make use of forward method
    pred = self.linear(x)
    return pred
    
``` 
``` 
#see the model structure 
torch.manual_seed(1)
model = LR(1, 1)
print(list(model.parameters()))
```
```
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
    
``` 
``` 

#each data point has an x and a y value
#we will be specifying x value by using randn
X = torch.randn(100, 1)*10 # returns a tensor filled with random numbers that are normally distributed
#it will have 100 rows and 1 column, they are y default centered around zero, to change that we multiply
#it by 10 to scale the values
y = X  #this creates an x = y line with normally distributed values, but this is not very challenging 
#y = X + 3*torch.randn(100, 1) #this adds noise in the data by shifting y value either upward or downward, so that the noise is also normally distributed 
#and since randn centers around zero, with a small standard deviation, so to make noise reasonable significant we multiply by 3
plt.plot(X.numpy(), y.numpy(), 'o')
plt.ylabel('y')
plt.xlabel('x')
#Now that we have created the dataset, we need to train a model to fit this dataset. Before going to that let's see how our previous definition of the model fits the data 
    
``` 
``` 
#re use the class we created earlier
class LR(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.linear = nn.Linear(input_size, output_size)
  def forward(self, x):
    pred = self.linear(x)
    return pred
    
``` 
``` 
    #we have random weights and bias assigned to model 
  torch.manual_seed(1)
model = LR(1, 1)
print(list(model.parameters()))
    
``` 
``` 
we can see what value is actually assigned to the parameters
[w, b] = model.parameters()
def get_params():
  return (w[0][0].item(), b[0].item()) this will return the two values as tuple
# we are doing this because these are the parameters we will update or try to change in order to FIT the data.
    
``` 
``` 
we draw the line defined by our random parameters and see how well it fits the data
def plot_fit(title):
  plt.title = title
  w1, b1 = get_params()
  x1 = np.array([-30, 30])
  y1 = w1*x1 + b1
  plt.plot(x1, y1, 'r')
  plt.scatter(X, y)
  plt.show()
    
``` 
``` 

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
We don't have to worry about the math when we code, it is just to know what is going behind the scene. 


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

The total error is the _sum of penalties associated with each point. 

### Concept of Cross-Entropy Loss
Mathematically we calculate the loss using the cross-entropy function. 
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
![Training Process](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/training_progress.png)

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
5. Softmax
etc

### Common Optimization Functions
1. Gradient Descent
2. Batch GD
3. SGD
4. GD with momentum
5. Adam
etc

More the no of hidden layers an nn has, the deeper it gets. We can combine many layers to obtain a very complex model. A deep NN is nothing but a multilayered perceptron. The number of hidden layers is called the depth of the neural network. 

We can visualize the concepts on http://playground.tensorflow.org/ this will give us some intuitive understanding of how the neural networks work. 

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

next, we use the previous code block to plot the data for visualization purpose 

```
# plotting data
def scatter_plot():
  plt.scatter(X[y==0, 0], X[y==0, 1])
  plt.scatter(X[y==1, 0], X[y==1, 1])
```
```
scatter_plot()
```
![Non linear data](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/nonlinear.png)

As we can see evidently the data set can not be separated by a single straight line and we will need a non-linear function to perform this task for us. to accomplish the task we need a deep neural network. In our perceptron model, we had an input layer and output layer but for our nn, as we saw a while ago, we need some quantity of hidden layers. and pass our input through the entire model to accomplish the forward pass.

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
Next, we initialize the model by passing in the number of nodes and initiating the model with random parameters.

```
torch.manual_seed(2)
model = Model(2, 4, 1) # we pass 2 input nodes, 4 nodes in hidden layer and output will be 1 node
print(list(model.parameters())) # we can see we have many more weight and bias parameters
```

For this task, we will replace our SDG optimizer with an Adam optimizer. In SGD we have to very mindful of the learning rate. A very small learning rate leads to slow convergence and very large learning might induce divergent behavior. On the other hand, the Adam optimizer algorithm does not maintain a single LR for all weight updates, it computes the adaptive learning rate. It is very popular with large models and datasets. It is a default for practice implementations, in research and academia. 

```
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # adam is a combination of two extensions of SDG adagrad and rmsprop , it is an adaptive algorithm
# use lr 0.1 
```

The training process remains the same 

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

now we need to plot the output, we will need to plot the nonlinear boundaries in form of contours. 

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
plot the decision boundary alongside scatter plot of original data points
 
```
plot_decision_boundary(X, y)
scatter_plot()
```
![Non-linear Boundries](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/nonlinearresult.png)

The white region is the -ve region, and a point lying in this region will have a value of 0, the point in the black region will have a prediction of 1
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

Until now we have performed binary classification on linear and non-linear binary data, now to increase the difficulty level we will increase the number of classes. Will will do so by classifying images belonging to more than just 2 classes.

### MNIST IMAGE RECOGNITION SECTION

The MNIST dataset is the "hello world" to image recognition. Its database of various handwritten digits. 

![MNIST data](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/mnist.png)

The MNIST dataset consists of 10 classes 0-9 in which we can classify numbers. They are typically 28 x 28 = 784 pixels. Hence as shown below, our input layer will have 784 nodes and output will have 10 nodes with a certain number of hidden layers with some other quantity of nodes.

![Model to classify mnist](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/mnist1.gif) | ![Model In Action](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/mnist2.gif)

Our goal is to train a model, capable to classify handwritten digits into proper classes. To do so first we introduce the concept of a training dataset and a test set. 

### Concept of Train Set and Test Set

In real-world applications, we are concerned to train machine learning algorithms to perform well on unseen data since this determines how well it will work when it is deployed in the real world. When a model is trained to fit training data, but not generalized to classify new data that it has never seen before, it fails to perform correctly. *The ability to correctly classify newly inputted data which don't have a label is called as Generalization *.   

The problem with training classifiers is that they tend to memorize the training set. Instead, to look for patterns or general features, it is just learning their labels. 

This is where Test Set comes in, the ability of a dataset to generalize is tested using a test set. 
For a good model, we want to have both training and test error to be adequately low. Ideally, we want the training error to be small, and we also want the gap between training error and test error to be small as well. We want our train model to effectively generalize our test data



![](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/traininerror.png)


1. Small training error corresponds to the problem of *underfitting*
2. Gap between test and training error growing larger corresponds to *over fitting*


### Underfitting

Underfitting is when the model is not provided with enough capacity for it to determine the data underlying trend. thus it is not able to fit the training set. 


### Overfitting 

If the model has a very high capacity it overfits the training data, it thus fails to accommodate new data.  

The following image depicts the problem of overfitting and underfitting and highlights the importance of the adequate model capacity for better generalization to unseen data. 

If a model performs well on training data and worse on test data, we say the model is overfitted. If a model is too over fitted we can do one on the following measures
1. reduce the depth or capacity of the network, number of hidden layers
2. reduce the number of nodes
3. reduce the number of epochs
4. dropouts
5. regularization 
etc



![Fitting Problems of data during training](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/fittingproblem.png)


#### Note on activation function for last layer of multiclass ploblem - Softmax Activation
One keynote to mention here! when dealing with the multiclass data we commonly make use of the softmax function in the output layer, rather than the sigmoid function. 

### Code Implementation MNIST Fully connected network
```
!pip3 install pillow==4.0.0
```
we will start by importing relevant packages, the torch vision package is a standard package that contains many standard types of data sets. 
it also provides common image transformations for preprocessing the dataset before feeding it into the neural network. The others which we are already familiar with.
```
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
```
Now we load the MNIST dataset into root dir, we initialize both training and test dataset. The transform argument implements any image manipulations we wish to apply. The following are the transforms we want to apply.
1. Transform the array into a float tensor, this will transform the array from value 0-255 to a float tensor of 0.0-1.0
2. Resize it to 28 x 28 size
3. Normalize the values, passing the mean and standard deviation of 0.5. normalization helps remove skewness and distortion. 
A training loader is what we use to specify the training batches, as passing 1 epoch with 60,000 training images is dataset will be very computationally expensive, so we pass the data in small batches, the batch size depends on the size of data and according to GPU. 
We shuffle the data to avoid getting stuck in local minima. 

```
transform = transforms.Compose([transforms.Resize((28,28)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))
                               ])
training_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
validation_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#print(training_dataset)
#print(validation_dataset)
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=100, shuffle=True) # define batch size 100
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = 100, shuffle=False) 
```

we can analyze the images by plotting them, for this we need to change it to a numpy array. Let's define a function for that

```
def im_convert(tensor):
  image = tensor.clone().detach().numpy() # we will clone, and detach and call numpy function
  # this returns a numpy array of 1x28x28 shape corresponding to channel x width x height.
  # to show the image we need a 28 x 28 x 1 array i.e width x height x channel.
  # hence we take a transpose
  image = image.transpose(1, 2, 0) # swaping axis
  image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5)) # now we denormalize the image by multiplying the SD and adding the mean
  image = image.clip(0, 1) # to ensure the range between min value 0 and max 1
  return image
```
Now, we need to iterate through data

```
dataiter = iter(training_loader) #create an object that allow us to itterate through training loader one element at a time. 
images, labels = dataiter.next() #access next item , grabs first batch into images and label
fig = plt.figure(figsize=(25, 4)) #create a fig for visualization

for idx in np.arange(20): # plot 20 images 
  ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])  # 2 rows, 10 items per row
  # x ticks remove grid lines
  plt.imshow(im_convert(images[idx])) # call the function we created
  ax.set_title([labels[idx].item()]) # add label in the title
  
```

now let's define the model
```
class Classifier(nn.Module): # define class as usual
    
    def __init__(self, D_in, H1, H2, D_out):  #we will have 2 hidden layers
        super().__init__() # for inheritence
        self.linear1 = nn.Linear(D_in, H1) # input layer
        self.linear2 = nn.Linear(H1, H2) # first hidden layer
        self.linear3 = nn.Linear(H2, D_out) #output layer
    #now from torch.nn.functional as F we use relu activation function
    def forward(self, x):
        x = F.relu(self.linear1(x))   # pass input through linear 1
        x = F.relu(self.linear2(x))  # pass output of layer 1 to layer 2
        x = self.linear3(x) # pass`output of layer 2 into 3rd output layer
        # we will not apply any activation function in last layer.
        return x
```

#### Note on Loss function 
Since we did not apply an activation function in the last layer we will get the raw output of the network. This is the consideration for our loss function. The Loss we will be using for the multiclass problem is nn.CrossEntropyLoss. We use CrossEntropyLoss whenever dealing with n class, it makes use of log probabilities, hence we pass the output of the network instead of the output of softmax activation function.

Cross entropy loss usses log_softmax + NLLLoss()   

Next, we initiate the model. we define the input-output size and hidden layer node size, this is a hyperparameter that needs to be tuned and you will develop intuition with practice.

```
model = Classifier(784, 125, 65, 10) # set input dimentions while initiatin the model 
model
```

Set up the loss criterion and optimizer as we have done previously.
```
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001) # we define a very low lr
```
Now we perform the training process, specify epochs and save data for analysis along the way

```
epochs = 15
running_loss_history = []
running_corrects_history = []
val_running_loss_history = []
val_running_corrects_history = []

for e in range(epochs): # itterate through batch
  # initaiate some variables to claculate and track errors
  running_loss = 0.0 # loss of each batch
  running_corrects = 0.0
  val_running_loss = 0.0
  val_running_corrects = 0.0
  
  for inputs, labels in training_loader: #for each epoch itterate through each batch
    inputs = inputs.view(inputs.shape[0], -1) # flatten the inputs, make it one dimentional
    outputs = model(inputs) # get predictions, make a forward pass
    loss = criterion(outputs, labels) # calculate the loss based on cross entropy criteria
    
    optimizer.zero_grad() #just to avoid accumulation of gradients
    loss.backward() #make a backward pass, back propogate
    optimizer.step() #update weights
    
    _, preds = torch.max(outputs, 1) # get the max score for every single image 
    # compare the top preds with actual label
    running_loss += loss.item()
    running_corrects += torch.sum(preds == labels.data) # get number of correct prediction for every single image

  else:
    with torch.no_grad():
      for val_inputs, val_labels in validation_loader:
        val_inputs = val_inputs.view(val_inputs.shape[0], -1)
        val_outputs = model(val_inputs) # get valudation prediction
        val_loss = criterion(val_outputs, val_labels) # calculate the validation loss
        
        _, val_preds = torch.max(val_outputs, 1)
        val_running_loss += val_loss.item()
        val_running_corrects += torch.sum(val_preds == val_labels.data)
      
    epoch_loss = running_loss/len(training_loader) # calculate epoch loss 
    epoch_acc = running_corrects.float()/ len(training_loader) 
    running_loss_history.append(epoch_loss) #append to loss history
    running_corrects_history.append(epoch_acc)
    
    val_epoch_loss = val_running_loss/len(validation_loader) 
    val_epoch_acc = val_running_corrects.float()/ len(validation_loader)
    val_running_loss_history.append(val_epoch_loss) # maintain validation loss hist
    val_running_corrects_history.append(val_epoch_acc) #maintain validation accuracy hist
    print('epoch :', (e+1)) # print current epoch number
    print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item())) # print training loss, and accuracy of network
    print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item())) # print validation loss, and accuracy of network
```
Let's visualize the training process.
Change the learning rate from 0.01 to 0.001 to understand more about the hyperparameter. 
Change the epochs from 12 to 15 to understand more about the hyperparameter. 

```
plt.plot(running_loss_history, label='training loss')
plt.plot(val_running_loss_history, label='validation loss')
plt.legend()
```
After tuning the code we are getting good results, the network performs well on the training set, but how well does it perform on the validation set? 
The results are very promising, the training error and validation error are both quite low. 
In case there are any extreme overfitting encounters, we have to tune hyperparameters and introduce some regularization techniques.  

We can plot the accuracy in pretty much the same way. 
```
plt.plot(running_corrects_history, label='training accuracy')
plt.plot(val_running_corrects_history, label='validation accuracy')
plt.legend()
```
Validation accuracy shows that the model is effectively creating predictions. We can also test this model by testing the images from the web.  

Load an image from the web and show it
```
import PIL.ImageOps
import requests
from PIL import Image

url = 'https://images.homedepot-static.com/productImages/007164ea-d47e-4f66-8d8c-fd9f621984a2/svn/architectural-mailboxes-house-letters-numbers-3585b-5-64_1000.jpg'
response = requests.get(url, stream = True) # make a get request to get data from web
img = Image.open(response.raw) 
plt.imshow(img)
```
our image is quite different from the images our model expects, so we will preprocess the image. 

```
img = PIL.ImageOps.invert(img) # invert the image
img = img.convert('1') this is an rgb image, we can convert this to binary image
img = transform(img) # we use the simmilar transforms we did on our original data
plt.imshow(im_convert(img)) # now lets show the data
```

Now we get the prediction on the image 

```
img = img.view(img.shape[0], -1) #reshape it to 1x784
output = model(img) # get pred
_, pred = torch.max(output, 1)
print(pred.item()) # print the prediction
```

Next for further visualization, we can make predictions on the validation set
```
dataiter = iter(validation_loader)  # grab images from validation loader
images, labels = dataiter.next() # grab images and labels
images_ = images.view(images.shape[0], -1) # to make predictions reshape them into 1 x 784
output = model(images_) # get output
_, preds = torch.max(output, 1) # get pred

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
  ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
  plt.imshow(im_convert(images[idx]))
  # if the label is currect we show it in green, else wise we show it in red, showing both 
  ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].item())), color=("green" if preds[idx]==labels[idx] else "red"))
```
This was a multi-class classification problem that we approached with a fully connected network. However, the default choice for image classification problems is the Convolutional neural network. The capacity of a Deep neural net will max out at a time and result in overfitting, no matter how much you tweak the hyperparameters. Another reason why we do not use FCN/ANN for image classification is that the MNIST is a grayscale image, however, in the real world, we have RGB images having a much larger size.  

Suppose we have an RGB image of size 480 x 360 pixels, the total number of input features will compute to 480x360x3 = 518400. This is computationally impossible for our machines. Hence we will shift our focus towards CNN

## Convolutional Neural Network
![CNN](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/cnn.jpeg)

CNNs have changed the way we classify images, as they are very effective at recognizing patterns in images by taking into account the spatial structure. ordinary NNS ignores the spatial relevance of pixels, i.e images being close together, etc. They require a lower quantity of parameters when compared to ANNs. There are two sections of CNN, namely the feature extraction section and the classification section.

### The layers of CNN
CNNs Comprise of Convolution layers, pooling layers, and fully connected layers. 

### The Convolution Operation
The name convolutional neural network comes from the convolution operations. These are key players in the CNNS. CNNS makes Image processing computationally manageable. All image pixels inside a convolutional layer are going to be processed by a convolutional filter called Kernel. Kernel matrices are small dimension matrices, We perform convolution operation by sliding the kernel at every location of the image. The amount by which we are shifting the kernel at every operation is known as a stride. A stride of 1 means that the filer will move one pixel at a time. the bigger the stride the smaller the corresponding feature map. This is called a feature map as the primary function of the CNN is to extract specific features. The kernel is used as a feature extractor. The more kernels we have more features we can learn. For conv operation, the depth of the kernel must match the depth of the image.
for a 480*360*3 image if we use a 3x3x3 kernel the total weight parameters will be 27, as compared to 518400 parameters of the FC network.  

![Convolution operation](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/rgbconv.gif)

### The Activation Function

After convolution which produced a 3d feature map, we apply the relu activation function. As we have seen that the real-world data is non-linear, while mathematically the convolution is a linear function. Hence to introduce nonlinearity we use a ReLu ( Rectified Linear Unit) operation. It converts -ve values to zero and produces only +ve values. We can also alternatively use tanh or sigmoid function. Relu is however more biologically inspired. 

![ReLu Activation Function](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/ReLu.png)

#### Note
The Sigmoid and Tanh functions are predisposed to a very common Neuran Network problem known as VANISHING GRADIENT. this refers to a decreased feed-forward gradient within Deep Neural Net.

### The Pooling Layer
Pooling layers continuously reduce no on parameters and computations. The pooling layer shrinks the image stack, by reducing the dimensionality of the feature map. Hence it reduced the computational complexity of the model while retaining the important information. Pooling helps to avoid overfitting. The different pooling operations include 
1. Sum
2. Average
3. Max Pooling

We will use the Max pooling operation, it returns the maximum output in a defined kernel region.  
![Pooling Operation](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/maxpooling.gif) 

The parameters associated with pooling layers are window size and stride. 


### The Fully Connected Layer
The convolution and pooling layers are used for feature extraction. Once the relevant features are extracted, then the Fully Connected layers come into action to make the actual classification. The FC layers assign a probability for input image belonging to some class. Each feature map is flattened before into a 1D array before it is fed to the fully connected network. The fully connected network produces the final probability. this is similar to what we saw in Deep Neural Network.  


In neural nets, lower layers correspond to simple image features while higher layers correspond to more sophisticated layers of the image. Before we go towards code, we can visualize the details in the following link https://www.cs.ryerson.ca/~aharley/vis/conv/ (flat.html). 

There are various CNN models, some famous ones are LeNet, AlexNet, ResNet, GoogleNet each preceding other in terms of performance and complexity. We will start with LeNet Model. The LeNet model consists of two conv layers each followed by a pooling layer for feature extraction. The classification section of LeNet comprise of two FC layers

### The Training Process

Following is the training procedure
1. Random values are initialized for all filters and parameters in CONV layers and all weights and bias in FC layers
2. Network receives an input goes through the length of NN, the feature extraction and classification layer
3. The Classification output is compared to the true label and a cross-entropy loss is calculated. 
4. The weights are updated to minimize the error using gradient descent. 
The only values that change during the training process are the values of the filter matrix in the Conv layer and weights in the FC layer. The process is very similar to what we have done until now. 


### Code Implementation MNIST CNN

We are replacing the DNN from the previous code with CNN. We are going to edit the older Classifier Class.   
nn.Conv2d takes 4 parameters, 
1. No of input channels
2. No of kernels (output channels)
3. kernel size
4. stride
FC layers defined as we did earlier using nn.linear

![No Of Features](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/convcode.png)

Enable GPU
```
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Note:
#15 epochs
#model = LeNet().to(device)
#images = images.to(device)
#labels = labels.to(device)
input,labels,val_inputs,val_label to devicetensor.cpu.clone

```
Define our model structure
```
class LeNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(1, 20, 5, 1) # 1 channel for gray scale, 20 channels, kernell size 5 and stride 1)
      self.conv2 = nn.Conv2d(20, 50, 5, 1) # 20 input, 50 filters,filer size 5 , stride 1
      self.fc1 = nn.Linear(4*4*50, 500) # in put channels 4*4*50 (50 channels) and output is 500
      #self.dropout1 = nn.Dropout(0.5) # dropout rate 0.5
      self.fc2 = nn.Linear(500, 10) # 500 input and output 10
    def forward(self, x):
      x = F.relu(self.conv1(x)) # pass through first layer and take relu
      x = F.max_pool2d(x, 2, 2) # pool layer 1
      x = F.relu(self.conv2(x)) # 2nd layer conv and relu
      x = F.max_pool2d(x, 2, 2) # 2nd pool layer
      x = x.view(-1, 4*4*50) # flatten before FC
      x = F.relu(self.fc1(x)) #pass to FC layer 1 and relu
      #x = self.dropout1(x) # apply dropout
      x = self.fc2(x) # no relu becauese multiclass classification problem
      return x
```

In case of over fitting, we can introduce dropouts. Dropouts are special layers. It randomly disconnects random nodes. This is to reduce the reliability of the network on any specific feature (node) so that all features are equally important when the network is in the learning phase. This ensures independent learning of all nodes. A different set of nodes are turned off at each update. Dropouts reduce generalization error. Typically dropouts are placed between layers having a large number of parameters. 
![Dropuots](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/dropout.png)



We have trained our very first CNN. We can move on to more complex CNN architectures and more complex data and more effective training techniques.  

## Introduction to CIFAR-10
We have seen MNIST Dataset which contains gray-scale single-channel images. The real-world data you will encounter will often be an RGB image with useful information in each channel so converting it to grayscale won't be a good idea. Cifar-10 is relatively harder to classify. It contains 50,000 training images and 10,000 Val images, containing 10 different classes. For each class color, size, magnification, possess are very diverse. Each image is 32x32 images. 

![CIFAR-10](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/cifar.png)

### Testing LeNet On Cifar-10
Earlier for MNIST class number was equal to digit number, for cifar we can map class number to class names by making a list
classes = ('plane', 'car','bird','cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

Since the input size changed from 28x28 the model inputs have to be modified accordingly
![CIFAR-10](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/cifarlenet.png)

Since the LeNet is not performing very well generalizing, we can try tuning some hyperparameters
the issues we are having are 
1. accuracy not being high enough
2. overfitting model

The model tuning is a case dependant procedure. We can do the following changes in our case
1. Change LR, training is slow as learning between consecutive epochs is very small so we can increase LR to 0.001
2. Adding more Conv Layers
3. Since we had overfitting, so we can reduce kernel size. from 5 to 3. Its a hyperparameter we need to tune
4. We can also include padding, in MNIST we ignored padding because our image ROI was not in the boundary region.
```
class LeNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
      self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
      self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
      self.fc1 = nn.Linear(4*4*64, 500)
      self.dropout1 = nn.Dropout(0.5)
      self.fc2 = nn.Linear(500, 10)
    def forward(self, x):
      x = F.relu(self.conv1(x))
      x = F.max_pool2d(x, 2, 2)
      x = F.relu(self.conv2(x))
      x = F.max_pool2d(x, 2, 2)
      x = F.relu(self.conv3(x))
      x = F.max_pool2d(x, 2, 2)
      x = x.view(-1, 4*4*64)
      x = F.relu(self.fc1(x))
      x = self.dropout1(x)
      x = self.fc2(x)
      return x
      
```
5. Additionally we can also use data augmentation techniques to increase the quantity of data for the training process. This is done by altering the image in useful ways. eg zoom, rotate crop, flip, darker shading, translation, noise, etc. This creates a variety of existing data and adds perspective. This can be achieved with the following transforms.

```
transform_train = transforms.Compose([transforms.Resize((32,32)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(10),
                                      transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])
```
Note the transforms are applied to only training data, not validation data, All available transforms and their details can be explored at the following link, https://pytorch.org/docs/stable/torchvision/transforms.html
 
We can also further perform regularization techniques, tune architecture complexity and parameters and you can play with the parameters on your own. Please share your results on SmartCityLab Twitter, Facebook, and LinkedIn pages. Post any errors and questions too, we will try to help you out. 


## State of the art Pre-trained Dataset
The final skill that we will acquire is the use of powerful Neural Nets for a highly complex task. A good model has two main contributors
1. Model architecture
2. Quality training
These two are high-cost parameters, this means that they lots of data and computational power. In a complex situation with time, data, and computation constraints we can use the technique called transfer learning. Just like learning one programming language can help to learn a new language. It is a skill that is great to have at your disposal. 

### When to use Transfer learning
1. when you don't have enough labeled data
2. effective model still exists

### AlexNet and VGG Model
We will use the AlexNet Model for Transfer learning. Its feature extraction section consist of 5 conv layers and the classifier section consists of 3 fully connected layers.
VGG-16 contains 13 Conv layers and 3 FC layers making 16 layers. These both are trained on ImageNet Dataset of 14 Million images, 20,000 classes. 
![AlexNet](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/AlexNet.png) |
![VGG(https://github.com/ShumailaAhmed/AI-Workshop/blob/main/VGG.png)


### Bees  and Ants Dataset
It contains images from Google search, contains 400 images. They are not visually very different, hence classification task is going to be very difficult. 

### Training procedure
For using pre-trained we 
1. Freeze the parameters in the feature extraction layer
2. We will train the FC layers and hence keep them dynamic
3. We will add a layer at end of the model with 2 nodes since we need to classify 2 objects 

This will help the model learn which features are more important for our dataset. 

### Code Transfer Learning

```from torchvision import models
```
```
!git clone https://github.com/jaddoescad/ants_and_bees.git
```
*see the folder structure*
Since the AlexNet model was trained on images of size 224x224 we change the input size. 
```
training_dataset = datasets.ImageFolder('ants_and_bees/train', transform=transform_train)
validation_dataset = datasets.ImageFolder('ants_and_bees/val', transform=transform)
```
The data set loader makes use of folder structure to assign classes. it will assign classes 0 to ants and 1 to bees. 

```
print(len(training_dataset))
print(len(validation_dataset))
```
we reduce the batch size to access it in smaller batches

```
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=20, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = 20, shuffle=False)
```
*Also change jitter transform to 1 and remove rotation transform*

Since we have only two classes we will change our class list
```
classes = ('ants', 'bees')
```
Now we can load the model 
```
model = models.vgg16(pretrained=True)
print(model)
```
now we have the model and recall we need to freeze the feature extraction layer to do so we will do as following

```
for param in model.features.parameters():
  param.requires_grad = False
```
Next, we need to change the last layer of the model.
1. We will access the 6th element of the classifier and access its input features. 
2. Create a new linear layer
3. Update the previous last layer with a new one
```
import torch.nn as nn

n_inputs = model.classifier[6].in_features
last_layer = nn.Linear(n_inputs, len(classes))
model.classifier[6] = last_layer
model.to(device)
print(model.classifier[6].out_features)
```

we can use the following training procedure
```
epochs = 15
running_loss_history = []
running_corrects_history = []
val_running_loss_history = []
val_running_corrects_history = []

for e in range(epochs):
  
  running_loss = 0.0
  running_corrects = 0.0
  val_running_loss = 0.0
  val_running_corrects = 0.0
  
  for inputs, labels in training_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    _, preds = torch.max(outputs, 1)
    running_loss += loss.item()
    running_corrects += torch.sum(preds == labels.data)

  else:
    with torch.no_grad():
      for val_inputs, val_labels in validation_loader:
        val_inputs = val_inputs.to(device)
        val_labels = val_labels.to(device)
        val_outputs = model(val_inputs)
        val_loss = criterion(val_outputs, val_labels)
        
        _, val_preds = torch.max(val_outputs, 1)
        val_running_loss += val_loss.item()
        val_running_corrects += torch.sum(val_preds == val_labels.data)
      
    epoch_loss = running_loss/len(training_loader.dataset)
    epoch_acc = running_corrects.float()/ len(training_loader.dataset)
    running_loss_history.append(epoch_loss)
    running_corrects_history.append(epoch_acc)
    
    val_epoch_loss = val_running_loss/len(validation_loader.dataset)
    val_epoch_acc = val_running_corrects.float()/ len(validation_loader.dataset)
    val_running_loss_history.append(val_epoch_loss)
    val_running_corrects_history.append(val_epoch_acc)
    print('epoch :', (e+1))
    print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item()))
    print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))
```
You can also try changing it to VGG

# APPENDIX I
## Hyperparameters to be mindful of
1. Learning rate
2. No of hidden layers
3. No of nodes in each hidden layers
4. no of epochs
for CNN
5. No of Conv and pooling layer
6. No of kernels/filters 
7. Size of kernels
8. Stride
9. Padding
10. pooling window size
11. Activation function
12. Dropout
etc

# APPENDIX II
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

Open google Colabotary  

```
import torch

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
 
 these 1D tensors behave like vectors, such that if we add these vectors, each homologous value is added, similar is the case with tensor multiplication and scalar multiplication.
 
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
  
# APPENDIX III
## Prerequisite Knowledge
1. Programming Basics
2. Python Basics
3. AI basics
4. Colab Basics


# We will love to answer more questions! We hope to see you in future trainings too!

![Questions](https://github.com/ShumailaAhmed/AI-Workshop/blob/main/questions.jpeg)

# Trainer Email: shumailaahmed@neduet.edu.pk
# https://twitter.com/smartcitylabned?s=09
# https://www.instagram.com/smartcitylab.ncai?r=nametag
# https://twitter.com/smartcitylabned?s=09

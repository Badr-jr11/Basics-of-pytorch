# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 10:27:33 2026

@author: AM4
"""

# Let's try to train a single neuron for a binary classification task

import pandas as pd  # pandas library for data manipulation
import matplotlib.pyplot as plt  # matplotlib for plotting graphs
import numpy as np  # numpy for working with vectors and matrices

# Load the data 
# df = pd.read_csv('https://archive.ics.uci.edu/ml/'
#     'machine-learning-databases/iris/iris.data', header=None)

df = pd.read_csv('data.csv')

# Inspect the data
print(df.head())

# Three columns are features, the fourth is the target variable (what we want to predict)

# Extract the target variable into a separate variable
y = df.iloc[:, 4].values

# Since the labels are strings, we need to convert them to numerical values
y = np.where(y == "Iris-setosa", 1, -1)

# Select two features for easier visualization
X = df.iloc[:, [0, 2, 1]].values

# Features in X, labels in y - let's visualize the task on a 2D plane
plt.figure()
plt.scatter(X[y==1, 0], X[y==1, 1], color='red', marker='o', label='Class 1')
plt.scatter(X[y==-1, 0], X[y==-1, 1], color='blue', marker='x', label='Class -1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Data Visualization')

# Proceed to creating the neuron
# Neuron function:
# value = w1*feature1 + w2*feature2 + w0 (bias)
# prediction = 1 if value >= 0
# prediction = -1 if value < 0

def neuron(w, x):
    """Single perceptron with step activation function"""
    if (w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[0]) >= 0:
        predict = 1
    else: 
        predict = -1
    return predict

# Test how it works (initialize weights arbitrarily)
w = np.array([0, 0.1, 0.4, 0.5])
print(neuron(w, X[0]))  # Output prediction for sample #1


# Now create the training procedure
# Weight update rule:
# w_new = w_old + eta * x * y (perceptron learning rule)

# Initialize weights randomly
w = np.random.random(4)
eta = 0.1  # learning rate
w_iter = []  # empty list to store weights for plotting later

for xi, target, j in zip(X, y, range(X.shape[0])):
    predict = neuron(w, xi)   
    # Update weights: target - predict is the error term
    w[1:] += eta * (target - predict) * xi  # update feature weights
    w[0] += eta * (target - predict)  # update bias
    # Save weights every 10th iteration for visualization
    if j % 10 == 0:
        w_iter.append(w.tolist())

# Calculate total errors
sum_err = 0
for xi, target in zip(X, y):
    predict = neuron(w, xi) 
    sum_err += abs(target - predict) / 2  # count misclassifications

print("Total errors: ", sum_err)


# Visualize the training process
xl = np.linspace(min(X[:, 0]), max(X[:, 0]))  # x-coordinate range for decision boundary

# Plot the data points first
plt.figure()
plt.scatter(X[y==1, 0], X[y==1, 1], color='red', marker='o', label='Class 1')
plt.scatter(X[y==-1, 0], X[y==-1, 1], color='blue', marker='x', label='Class -1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Training Process: Decision Boundary Evolution')

# Iterate through saved weights and plot corresponding decision boundaries
for i, w in zip(range(len(w_iter)), w_iter):
    # Decision boundary equation: w1*x1 + w2*x2 + w0 = 0
    # => x2 = -(w1*x1 + w0) / w2
    yl = -(xl * w[1] + w[0]) / w[2]
    plt.plot(xl, yl, alpha=0.5)  # plot decision boundary
    plt.text(xl[-1], yl[-1], str(i), dict(size=10, color='gray'))  # label iteration number
    plt.pause(0.5)  # pause for animation effect
    
plt.text(xl[-1]-0.3, yl[-1], 'END', dict(size=14, color='red'))
plt.legend()
plt.show()

# Let's try to use pytorch to create the neuron
import torch
import torch.nn as nn

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape: [n_samples, 1]


input_dim=3

# one neuron  in pytorch is single linear layer: output = w1*x1 + w2*x2 + bias
model = nn.Linear(input_dim, 1, bias=True)
print(model.weight)
print('----------------')
print(model.bias)


eta = 0.01  # learning rate

# Manual weight update using perceptron rule 
w_iter = []  # store weights for visualization

for xi, target, j in zip(X_tensor, y_tensor, range(X_tensor.shape[0])):

    # Forward pass
    prediction = model(xi)
    # Apply step function: sign of output determines class
    prediction = torch.where(prediction >= 0, torch.tensor(1.0), torch.tensor(-1.0))
    
    # weight update: w_new = w_old + eta * (target - prediction) * x
    # access weights and bias directly
    with torch.no_grad():
        error = target - prediction
        # Update weights (gradient of loss w.r.t. weights is -error * x for perceptron)
        model.weight += eta * error * xi
        model.bias += eta * error
    
    # Save weights every 10 iterations
    if j  % 10 == 0:
        w = [model.bias.item(), 
             model.weight[0, 0].item(), 
             model.weight[0, 1].item()]
        w_iter.append(w)

# Calculate final errors
with torch.no_grad():
    predictions = model(X_tensor)
    prediction = torch.where(predictions >= 0, torch.tensor(1.0), torch.tensor(-1.0))
    total_errors = (y_tensor != prediction).sum().item()/2
    print(f"Total errors (PyTorch): {total_errors}")

# Visualization (same as original)
xl = np.linspace(min(X[:, 0]), max(X[:, 0]))

plt.figure()
plt.scatter(X[y==1, 0], X[y==1, 1], color='red', marker='o', label='Class 1')
plt.scatter(X[y==-1, 0], X[y==-1, 1], color='blue', marker='x', label='Class -1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('PyTorch: Decision Boundary Evolution')

for i, w in enumerate(w_iter):
    # Decision boundary: w[1]*x1 + w[2]*x2 + w[0] = 0
    yl = -(xl * w[1] + w[0]) / w[2]
    plt.plot(xl, yl, alpha=0.5)
    plt.text(xl[-1], yl[-1], str(i), dict(size=10, color='gray'))
    plt.pause(0.5)

plt.text(xl[-1]-0.3, yl[-1], 'END', dict(size=14, color='red'))
plt.legend()
plt.show()

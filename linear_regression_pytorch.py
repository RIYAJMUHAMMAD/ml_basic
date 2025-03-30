import numpy as np
import pandas as pd

import torch
import plotly.express as px
from torch.utils.data import TensorDataset, DataLoader

train_data = pd.read_csv("train (1).csv")
train_data = train_data.dropna()

X_train = train_data["x"].values
y_train = train_data["y"].values

def standardize_data(X_train):
    mean = np.mean(X_train, axis =0)
    std = np.std(X_train, axis =0)

    X_train = (X_train-mean)/std
    return X_train

X_train = standardize_data(X_train) 
X_train = np.expand_dims(X_train, axis =-1)

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)

class PytorchRegression:

    def __init__(self, learning_rate, convergence_tol = 1e-6):

        self.learning_rate = learning_rate
        self.convergence_tol = convergence_tol
        self.W = None
        self.b = None


    def initialize_parameters(self, n_features):

        self.W = torch.randn(n_features, requires_grad = False)*0.01
        self.b = torch.tensor(0.0, requires_grad = False)

    def forward(self, X):

        return torch.matmul(X, self.W) + self.b

    def compute_cost(self, predictions):
        m = len(predictions)
        cost = torch.sum(torch.square(predictions-self.y))/ (2*m)
        return cost.item()

    def backward(self, predictions):
        m = len(predictions)
        self.dW = torch.matmul(self.X.T, (predictions - self.y)) / m
        self.db = torch.sum(predictions- self.y)/m

    
    def fit(self, X, y, iterations, plot_cost = True):

        assert isinstance(X, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert X.shape[0] == y.shape[0]
        assert iterations>0

        self.X = X
        self.y = y
        self.initialize_parameters(X.shape[1])
        costs = []

        for i in range(iterations):
            predictions = self.forward(X)

            cost = self.compute_cost(predictions)

            self.backward(predictions)
            self.W -= self.learning_rate* self.dW
            self.b -= self.learning_rate* self.db

            costs.append(cost)

            if i% 100 == 0:
                print(f"Iterations: {i}, Cost: {cost}")

            if i > 0 and abs(costs[-1] - costs[-2]) < self.convergence_tol:
                print(f'Converged after {i} iterations.')
                break
        
        if plot_cost:
            fig = px.line(y=costs, title="Cost vs Iterations", template="plotly_dark")
            fig.update_layout(
                title_font_color="#41BEE9",
                xaxis=dict(color="#41BEE9", title="Iterations"),
                yaxis=dict(color="#41BEE9", title="Cost")
            )
            fig.show()
    
    def predict(self, X):
        return self.forward(X)

# Initialize and train the model
lr = PytorchRegression(0.01)
lr.fit(X_train_tensor, y_train_tensor, 10000)


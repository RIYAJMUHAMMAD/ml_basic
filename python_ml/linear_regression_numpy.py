import numpy as np
import pandas as pd
import plotly.express as px


train_data = pd.read_csv("train (1).csv")
train_data = train_data.dropna()

X_train = train_data['x'].values
y_train = train_data['y'].values

def standardize_data(X_train):
    """
    Standardizes the input data using mean and standard deviation.

    Parameters:
        X_train (numpy.ndarray): Training data.
        X_test (numpy.ndarray): Testing data.

    Returns:
        Tuple of standardized training and testing data.
    """
    # Calculate the mean and standard deviation using the training data
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Standardize the data
    X_train = (X_train - mean) / std
    
    return X_train

X_train = standardize_data(X_train)

X_train = np.expand_dims(X_train, axis=-1)

# Set testing data and target
#X_test = test_data['x'].values
#y_test = test_data['y'].values
class Regression:
    def __init__(self, learning_rate, convergence_tol=1e-6):
        self.learning_rate = learning_rate
        self.convergence_tol= convergence_tol
        self.W = None
        self.b = None

    def initialize_parameters(self, n_features):
        scale = np.sqrt(2/(n_features+1))
        self.W = np.random.randn(n_features) * scale
        self.b = 0

    def forward(self, X):
        return np.dot(X, self.W) + self.b

    def compute_cost(self, predictions):

        m = len(predictions)
        cost = np.sum(np.square(predictions - self.y))/ (2*m)
        return cost

    def backward(self, predictions):
        m = len(predictions)
        self.dW = np.dot(self.X.T,predictions- self.y) / m
        self.db = np.sum(predictions - self.y) /m

    def fit(self, X, y, iterations, plot_cost = True):

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
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
            self.W -= self.learning_rate * self.dW
            self.b -= self.learning_rate * self.db
            costs.append(cost)

            if i%100 == 0:
                print(f"Iteration: {i}, Cost:{cost}")

            if i>0 and abs(costs[-1]- costs[-2]) < self.convergence_tol:
                print(f'Converged after {i} iterations.')
                break


        if plot_cost:
            fig = px.line(y=costs, title= "cost vs iterations", 
            template = "plotly_dark")
            fig.update_layout(
                title_font_color = "#41BEE9",
                xaxis = dict(color = "#41BEE9", title = "Iterations"),
                yaxis = dict(color = "#41BEE9", title = "Cost")
            )
            fig.show()

    def predict(self, X):
        return self.forward(X)


lr = Regression(0.01)
lr.fit(X_train, y_train, 10000)
            



import numpy as np

# defines the compute cost function with input pararmeters:
# X: input/feature matrix of size (m,n) where m is number of training examples and n is the number of features
# y: output/target variable we are trying to predict of size (m,1)
# theta: parameter/weight vector of size (n,1) this is how we predict y using X
# trying to find theta values that minimize the cost function below
def compute_cost(X, y, theta):
    # Get number of training examples (m) using the size of y
    m = y.size
    
    # Initialize the cost variable
    cost = 0
    
    # Calculate the cost using mean squared formula
    # Cost is the sum of squared differences between the predicted (X * theta) and actual values (y)
    # Divide by twice the number of training examples
    cost = np.sum((np.dot(X, theta) - y) ** 2) / (2 * m)

    return cost


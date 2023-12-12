import numpy as np
from computeCost import *

# defines the gradient descent function with input pararmeters:
# X: input/feature matrix of size (m,n) where m is number of training examples and n is the number of features
# y: output/target variable we are trying to predict of size (m,1)
# theta: parameter/weight vector of size (n,1) this is how we predict y using X
# alpha: learning rate (independent variable) controls how much we update theta with each iteration
# num_iters: number of iterations (independent variable) the gradient descent function will  run for
# trying to find theta values that minimize the cost function

def gradient_descent(X, y, theta, alpha, num_iters):
    # Get number of training examples (m) from the size of the target variable (y)
    m = y.size
    # Create an array to store the cost of each iteration
    J_history = np.zeros(num_iters)

    # Loop through specified number of iterations
    for i in range(0, num_iters):

        # Calculate error by subtracting actual (y) from calculated (X * theta)
        error = np.dot(X, theta).flatten() - y
        # Update theta using gradient descent
        # This formula is derived from the gradient of the cost function
        # The learning rate (alpha) controls how large of a step we'll move along the gradient to update theta
        # The sum calculates element by element the error between calculated output and y multiplied by input (X)
        theta -= (alpha/m) * np.sum(X * error[:, np.newaxis], 0)

        # store the histrory of cost J to track if it's decreasing
        J_history[[i]] = compute_cost(X, y, theta)

    return theta, J_history


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(0, num_iters):
        error = np.dot(X, theta).flatten() - y
        theta -= (alpha/m) * np.sum(X * error[:,np.newaxis],0)

        J_history[[i]] = compute_cost(X, y, theta)

    return theta, J_history
    


import numpy as np

# defines the normal equation function to "solve" linear regression with input pararmeters:
# X: input/feature matrix of size (m,n) where m is number of training examples and n is the number of features
# y: output/target variable we are trying to predict of size (m,1)
def normal_eqn(X, y):
    # Initializes theta vector with zeros to be the size of the number of features in X
    theta = np.zeros((X.shape[1], 1))

    # Transpose the feature matrix X (turn rows into columns and vice versa)
    Xt = np.transpose(X)
    # Use the normal equation to compute optimat theta
    # theta = (X^T * X)^(-1) * X^T * y
    # np.linalg.pinv(Xt.dot(X)) computes the pseado-inverse of Xt.dot(X)
    # result multiplied by Xt and y to determine optimal theta
    theta = np.linalg.pinv(Xt.dot(X)).dot(Xt).dot(y)


    return theta

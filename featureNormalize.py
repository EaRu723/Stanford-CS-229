import numpy as np

# funciton no normalize feature vector
# input (X) feature matrix
def feature_normalize(X):
    # Get the number of features
    n = X.shape[1]
    # initialize features
    X_norm = X
    mu = np.zeros(n)
    sigma = np.zeros(n)

    # calculate the mean of each feature
    mu = np.mean(X,0)
    # calculate the STD of each feature
    sigma = np.std(X,0, ddof = 1)
    # Normalize the features by subtracting the mean and dividing the standard deviation
    X_norm = (X - mu) /sigma

    return X_norm, mu, sigma

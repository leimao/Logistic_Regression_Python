
import numpy as np

def sigmoid(Z):

    # Sigmoid function
    # g(Z) = 1 / (1 + e^{-Z})
    # Z: shape [m,]

    # Trick to prevent possible overflow of np.exp(-Z)
    # Output will never to be exact 0
    idx = (Z < -500)
    Z[idx] = -500

    return 1.0 / (1.0 + np.exp(-Z))

class LogisticRegressor(object):

    def __init__(self, alpha, c, T = 1000, random_seed = 0, intercept = True):

        # Initialize Logistic Regression
        # alpha: learning rate.
        # c: L2 regularization strength.

        self.alpha = alpha
        self.c = c
        self.T = T
        self.random_seed = random_seed
        self.intercept = intercept


    def fit(self, X, y):

        np.random.seed(self.random_seed)

        y = y.flatten()

        if self.intercept == True:
            X = np.insert(X, 0, np.ones(X.shape[0]), axis = 1)

        # Weights initialization
        self.theta = np.random.normal(0, 0.1, X.shape[1])

        losses = list()
        losses.append(self.logistic_regression_loss(X = X, y = y, theta = self.theta, c = self.c))

        for i in range(self.T):

            self.theta = self.logistic_regression_weight_update(X = X, y = y, theta = self.theta, alpha = self.alpha, c = self.c)
            loss = self.logistic_regression_loss(X = X, y = y, theta = self.theta, c = self.c)
            losses.append(loss)

        losses = np.array(losses)

        return losses


    def logistic_regression_loss(self, X, y, theta, c):

        # Loss function for logistic regression
        # X: input feature matrix, shape [m,n].
        # y: input target value, shape [m,1].
        # c: L2 regularization strength.

        h = sigmoid(Z = X.dot(theta))

        # Trick to prevent h is exact 1 for np.log(1 - h)
        idx = (h == 1.0)
        h[idx] = 1.0 - 1e-15

        loss_mean = 1.0 / X.shape[0] * (-np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))) + c * np.sum((theta ** 2))

        return loss_mean


    def logistic_regression_weight_update(self, X, y, theta, alpha, c):
    
        # Weight update of gradient descent for logistic regression
        # X: input feature matrix, shape [m,n].
        # y: input target value, shape [m,1].
        # alpha: learning rate.
        # c: L2 regularization strength.
        # theta: parameters for features in X, shape [n+1,1].

        # Calculate activated values
        h = sigmoid(Z = X.dot(theta))
        # Update weigths
        theta += alpha * ((1.0 / X.shape[0] * (y - h).dot(X)) - 2 * c * theta)

        return theta


    def predict(self, X, threshold = 0.5):

        # threshold: predict 1 if above threshold.

        if self.intercept == True:
            X = np.insert(X, 0, np.ones(X.shape[0]), axis = 1)

        probabilities = sigmoid(Z = np.dot(X, self.theta))

        y_predicted = (probabilities > threshold).astype(int)

        return y_predicted
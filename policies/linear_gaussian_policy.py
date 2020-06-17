import numpy as np


class LinearGaussianPolicy:

    def __init__(self, weights=None, noise=None):
        if weights is not None:
            self.weights = weights
            self.output, self.input = self.weights.shape
        if noise is not None and isinstance(noise, (int,  float, complex)):
            noise = np.diag(np.ones(self.output)*noise)
        self.noise = noise

    def get_weights(self):
        return self.weights

    def set_weights(self, weights, noise=None):
        self.weights = weights
        self.output, self.input = self.weights.shape
        if noise is not None and isinstance(noise, (int, float, complex)):
            noise = np.diag(np.ones(self.output)*noise)
        self.noise = noise

    def _add_noise(self):
        noise = np.random.multivariate_normal(np.zeros(self.output), self.noise, 1).T
        return noise

    def act(self, X, stochastic=True):
        X = X.reshape(self.input, 1)
        y = np.dot(self.weights, X)
        if self.noise is not None and stochastic:
            y += self._add_noise()
        return y

    def step(self, X, stochastic=False):
        return None, self.act(X, stochastic), None, None

    def compute_gradients(self, X, y, diag=False):
        X = np.array(X).reshape(self.input, 1)
        y = np.array(y).reshape(self.output, 1)
        mu = np.dot(self.weights, X)
        if diag:
            return np.diag((np.dot(np.linalg.inv(self.noise), np.dot((y - mu), X.T))))
        else:
            return (np.dot(np.linalg.inv(self.noise), np.dot((y - mu), X.T))).flatten()

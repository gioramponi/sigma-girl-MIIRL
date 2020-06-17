import numpy as np


class GradientDescent(object):

    '''
    Abstract class
    '''

    def initialize(self, x0):
        self.x = x0

    def update(self, dx):
        pass


class VanillaGradient(GradientDescent):

    def __init__(self, learning_rate, ascent=False):
        self.leaning_rate = learning_rate
        self.ascent = ascent

    def update(self, dx):
        if self.ascent:
            self.x += self.leaning_rate * dx
        else:
            self.x -= self.leaning_rate * dx

        return self.x


class Adam(GradientDescent):

    '''
    Kingma, Diederik, and Jimmy Ba.
    "Adam: A method for stochastic optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    '''

    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8, use_correction=False, ascent=False, clip=None, normalize_sum=False):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.use_correction = use_correction
        self.ascent = ascent
        self.clip = clip
        self.normalize_sum = normalize_sum

    def initialize(self, x0):
        self.x = x0
        self.m = 0
        self.v = 0
        self.t = 0

    def update(self, dx):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * dx
        self.v = self.beta2 * self.v + (1 - self.beta2) * (dx ** 2)

        if self.use_correction:
            m = self.m / (1 - self.beta1 ** self.t)
            v = self.v / (1 - self.beta2 ** self.t)
        else:
            m = self.m
            v = self.v

        if self.ascent:
            self.x += self.learning_rate * m / (np.sqrt(v) + self.eps)
        else:
            self.x -= self.learning_rate * m / (np.sqrt(v) + self.eps)

        if self.clip is not None:
            self.x = np.clip(self.x, self.clip[0], self.clip[1])

        if self.normalize_sum and np.sum(self.x) > 1:
            self.x /= np.sum(self.x)

        return self.x

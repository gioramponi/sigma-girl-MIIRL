import numpy as np
from utils import feature_expectations


class RelativeEntropyIRL(object):
    eps = 1e-24

    def __init__(self,
                 reward_features,
                 reward_random,
                 gamma,
                 horizon,
                 trajectories_expert=None,
                 trajectories_random=None,
                 n_states=None,
                 n_actions=None,
                 learning_rate=0.01,
                 max_iter=100,
                 type_='state',
                 gradient_method='linear',
                 evaluation_horizon=100):

        # transition model: tensor (n_states, n_actions, n_states)

        self.reward_features = reward_features
        self.reward_random = reward_random
        self.trajectories_expert = trajectories_expert
        self.trajectories_random = trajectories_random
        self.gamma = gamma
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        if not type_ in ['state', 'state-action']:
            raise ValueError()
        self.type_ = type_

        if not gradient_method in ['linear', 'exponentiated']:
            raise ValueError()
        self.gradient_method = gradient_method

        self.evaluation_horizon = evaluation_horizon

        self.n_states, self.n_actions = n_states, n_actions
        self.n_features = reward_features.shape[2]

    def fit(self, verbose=False):

        #Compute features expectations
        expert_feature_expectations = feature_expectations(self.reward_features, self.gamma)
        random_feature_expectations = feature_expectations(self.reward_random, self.gamma)
        return reirl(expert_feature_expectations, random_feature_expectations, max_iter=self.max_iter,
                     learning_rate=self.learning_rate, verbose=verbose)

        # expert_feature_expectations
        # n_random_trajectories = self.reward_random.shape[0]
        # importance_sampling = np.zeros(n_random_trajectories)
        #
        # #Weights initialization
        # w = np.ones(self.n_features) / self.n_features
        # #Gradient descent
        # for i in range(self.max_iter):
        #     if verbose:
        #         print('Iteration %s/%s' % (i + 1, self.max_iter))
        #
        #     for j in range(n_random_trajectories):
        #         importance_sampling[j] = np.dot(random_feature_expectations[j], w)
        #     importance_sampling -= np.max(importance_sampling)
        #     importance_sampling = np.exp(importance_sampling)/np.sum(np.exp(importance_sampling), axis=0)
        #     weighted_sum = np.sum(
        #         np.multiply(np.array([importance_sampling, ] * random_feature_expectations.shape[1]).T,
        #                     random_feature_expectations), axis=0)
        #
        #     w += self.learning_rate * (expert_feature_expectations_mean - weighted_sum)
        #     # One weird trick to ensure that the weights don't blow up the objective.
        #     w = w / np.linalg.norm(w, keepdims=True)
        # return w


def reirl(expert_feature_expectations, random_feature_expectations, max_iter=100, learning_rate=0.01, verbose=False):
    # Compute features expectations
    expert_feature_expectations_mean = np.mean(expert_feature_expectations, axis=0)

    n_random_trajectories = int(len(random_feature_expectations))
    importance_sampling = np.zeros(n_random_trajectories)

    # Weights initialization
    n_features = expert_feature_expectations_mean.shape[-1]
    w = np.ones(n_features) / n_features

    # Gradient descent
    for i in range(max_iter):

        if verbose:
            print('Iteration %s/%s' % (i + 1, max_iter))

        for j in range(n_random_trajectories):
            importance_sampling[j] = np.exp(np.dot(random_feature_expectations[j], w))
        importance_sampling /= np.sum(importance_sampling, axis=0)
        weighted_sum = np.sum(np.multiply(np.array([importance_sampling, ] * random_feature_expectations.shape[1]).T,
                                          random_feature_expectations), axis=0)

        w += learning_rate * (expert_feature_expectations_mean - weighted_sum)

        # One weird trick to ensure that the weights don't blow up the objective.
        w = w / np.linalg.norm(w, keepdims=True)

    w /= np.linalg.norm(w, ord=1, keepdims=True)
    return w




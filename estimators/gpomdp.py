import numpy as np


class GPOMDP():
    def __init__(self, gradient, gamma, rewards, weights=None, probs=[1]):
        self.gradients = gradient
        self.gamma = gamma
        self.weights = weights
        self.rewards = rewards
        self.probs = probs

    def eval_gpomdp(self):
        if self.weights == None:
            self.weights = 1
        if len(self.probs) == 1:
            self.probs = np.ones((self.gradients.shape[0], self.gradients.shape[1]))
        sn = (np.mean(self.probs, axis=0) + 1.e-10)
        discount_factor_timestep = np.power(self.gamma * np.ones(self.gradients.shape[1]),
                                            range(self.gradients.shape[1]))  # (T,)
        discounted_return = discount_factor_timestep[np.newaxis, :, np.newaxis] * self.rewards  # (N,T,L)
        gradient_est_timestep = np.cumsum(self.gradients, axis=1) * self.probs[:, :, np.newaxis, np.newaxis] / \
                                sn[np.newaxis, :, np.newaxis, np.newaxis]  # (N,T,K, 2)
        gradient_est_timestep2 = np.cumsum(self.gradients, axis=1) ** 2 * self.probs[:, :, np.newaxis, np.newaxis] / \
                                 sn[np.newaxis, :, np.newaxis, np.newaxis]  # (N,T,K, 2)
        baseline_den = np.mean(gradient_est_timestep2 + 1.e-10, axis=0)  # (T,K, 2)
        baseline_num = np.mean(
            gradient_est_timestep2[:, :, :, :, np.newaxis] * discounted_return[:, :, np.newaxis, np.newaxis, :],
            axis=0)  # (T,K,2,L)
        baseline = baseline_num / baseline_den[:, :, :, np.newaxis]  # (T,K,2,L)
        gradient = np.mean(
            np.sum(gradient_est_timestep[:, :, :, :, np.newaxis] * (discounted_return[:, :, np.newaxis, np.newaxis, :] -
                                                                    baseline[np.newaxis, :, :]), axis=1),
            axis=0)  # (K,2,L)
        return gradient

    def eval_gpomdp_discrete(self):
        if self.weights == None:
            self.weights = 1
        if len(self.probs) == 1:
            self.probs = np.ones((self.gradients.shape[0], self.gradients.shape[1]))
        discount_factor_timestep = np.power(self.gamma * np.ones(self.gradients.shape[1]),
                                            range(self.gradients.shape[1]))  # (T,)
        discounted_return = discount_factor_timestep[np.newaxis, :, np.newaxis] * self.rewards  # (N,T,L)
        gradient_est_timestep = np.cumsum(self.gradients, axis=1) * self.probs[:, :, np.newaxis]  # (N,T,K, 2)
        baseline_den = np.mean(gradient_est_timestep ** 2 + 1.e-10, axis=0)  # (T,K, 2)
        baseline_num = np.mean(
            (gradient_est_timestep ** 2)[:, :, :, np.newaxis] * discounted_return[:, :, np.newaxis, :],
            axis=0)  # (T,K,2,L)
        baseline = baseline_num / baseline_den[:, :, np.newaxis]  # (T,K,2,L)
        gradient = np.mean(
            np.sum(gradient_est_timestep[:, :, :, np.newaxis] * (discounted_return[:, :, np.newaxis, :] -
                                                                 baseline[np.newaxis, :, :]), axis=1),
            axis=0)  # (K,2,L)
        return gradient

import numpy as np

class GPOMDP():
    def __init__(self, episode_length, reward_space, opt=None, step_size=1e-4):
        self.episode_length = episode_length
        self.reward_space = reward_space

        if opt is None:
            self.lr = lambda _: step_size
        else:
            self.lr = lambda grad: opt.update(grad, step_size)

    def estimate_gpomdp_gradients(self, gradients, sampled_phi):
        num_episodes_sampled = sampled_phi.shape[0]
        num_params = gradients.shape[2]

        # GPOMDP optimal baseline computation
        cum_gradients = np.transpose(np.tile(gradients, (self.reward_space, 1, 1, 1)), axes=[1, 2, 3, 0]).cumsum(axis=1)
        phi = np.transpose(np.tile(sampled_phi, (num_params, 1, 1, 1)), axes=[1, 2, 0, 3])
        num = (cum_gradients ** 2 * phi).sum(axis=0)
        den = (cum_gradients ** 2).sum(axis=0) + 1e-10
        baseline = num / den

        # GPOMDP objective function gradient estimation
        baseline = np.tile(baseline, (num_episodes_sampled, 1, 1, 1))
        sum_sampled_phi = sampled_phi.sum(axis=1)
        estimated_gradients = (cum_gradients * (phi - baseline)).sum(axis=1).mean(axis=0)
        print('Sampled features expectation:', sum_sampled_phi.mean(axis=0))
        print('Estimated gradients norm: ', np.linalg.norm(estimated_gradients))
        print('Max estimated gradient: ', estimated_gradients.max())
        return estimated_gradients

    def fit_model(self, pi, sampled_phi, X_dataset, y_dataset):

        steps = len(X_dataset)
        num_episodes = steps // self.episode_length

        # computing the gradients of the logarithm of the policy wrt policy parameters
        episode_gradients = []
        for step in range(steps):
            step_layers = pi.compute_gradients([X_dataset[step]], [y_dataset[step]])
            step_gradients = []
            for layer in step_layers:
                step_gradients.append(layer.ravel())
            episode_gradients.append(np.concatenate(step_gradients))
        gradients = []
        for episode in range(num_episodes):
            base = episode * self.episode_length
            gradients.append(episode_gradients[base: base + self.episode_length])
        gradients = np.array(gradients)

        estimated_gradients = self.estimate_gpomdp_gradients(gradients, sampled_phi)
        # model update
        lr = self.lr(estimated_gradients[:, 0])
        weights = pi.get_weights()
        updated_weights = weights + lr * estimated_gradients[:, 0]
        pi.set_weights(updated_weights.tolist())
        return updated_weights

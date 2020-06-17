import numpy as np
import pickle
from envs.continuous_mountaincar import ContinuousMountainCar
from joblib import Parallel, delayed
from scipy import optimize
import cvxpy as cvx


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


    def fit(self, verbose=False, gradient=False, num_iters=10):
        #Compute features expectations
        expert_feature_expectations = np.mean(feature_expectations(self.reward_features, self.gamma), axis=0)
        print(expert_feature_expectations)
        print(self.reward_features.shape)
        random_feature_expectations = feature_expectations(self.reward_random, self.gamma)
        print(random_feature_expectations.sum(axis=0))
        n_random_trajectories = self.reward_random.shape[0]
        importance_sampling = np.zeros(n_random_trajectories)
        if gradient:
            #Weights initialization
            w = np.ones(self.n_features) / self.n_features
            print(w.shape)
            #Gradient descent

            for i in range(self.max_iter):

                if verbose:
                    print('Iteration %s/%s' % (i + 1, self.max_iter))

                for j in range(n_random_trajectories):
                    importance_sampling[j] = np.dot(random_feature_expectations[j], w)
                importance_sampling -= np.max(importance_sampling)
                importance_sampling = np.exp(importance_sampling)/np.sum(np.exp(importance_sampling), axis=0)
                weighted_sum = np.sum(
                    np.multiply(np.array([importance_sampling, ] * random_feature_expectations.shape[1]).T,
                                random_feature_expectations), axis=0)

                w += self.learning_rate * (expert_feature_expectations - weighted_sum)
                # One weird trick to ensure that the weights don't blow up the objective.
                w = w / np.linalg.norm(w, keepdims=True)
            w /= w.sum()
            return w
        else:
            def obj_func(w):
                objective = np.dot(expert_feature_expectations, w)
                try:
                    for j in range(n_random_trajectories):
                        importance_sampling[j] = np.dot(random_feature_expectations[j], w)
                    m = np.max(importance_sampling)
                    log_z = m + np.log(np.exp(importance_sampling - m).sum())
                    objective -= log_z
                except Exception as err:
                    objective += 150
                return - objective

            bound = (0, np.inf)
            bounds = [bound] * self.n_features
            evaluations = []
            i = 0
            while i < num_iters:
                x0 = np.random.uniform(0, 1, self.n_features)
                x0 = x0 / np.sum(x0)
                try:
                    res = optimize.minimize(obj_func,
                                            x0,
                                            method='SLSQP',
                                            bounds=bounds,
                                            options={'ftol': 1e-8, 'disp': verbose})
                except Exception as err:
                    print("Error")
                if res.success:
                    evaluations.append([res.x, obj_func(res.x)])
                    i += 1
            evaluations = np.array(evaluations)
            min_index = np.argmin(evaluations[:, 1])
            x, y = evaluations[min_index, :]
            print("WEIGHTS",  x / np.sum(x))
            return x / np.sum(x)


def feature_expectations(rewards, gamma):
    discount_factor_timestep = np.power(gamma * np.ones(rewards.shape[1]),
                                        range(rewards.shape[1]))
    discounted_return = discount_factor_timestep[np.newaxis, :, np.newaxis] * rewards
    reward_est_timestep = np.sum(discounted_return, axis=1)
    return reward_est_timestep


def run(id,seed):
    np.random.seed(seed)
    weights = np.zeros((20,15))
    # path = '/Users/giorgiaramponi/Documents/GitHub/multipleIntentIRL/logs/cont_gridworld/gpomdp3/'
    path = ''
    # path = '/Users/giorgiaramponi/Documents/GitHub/multipleIntentIRL/logs/cont_mountaincars/gpomdp3/'
    # path = '/Users/giorgiaramponi/Documents/GitHub/multipleIntentIRL/logs/cont_microcell/gpomdp3/'
    _, actions, rewards, states = pickle.loads(
        open(path+'cont_mountaincars/gpomdp3/dataset_%s/trajectories.pkl' % str(id),'rb').read())

    for i, j in enumerate([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
        # param_r = pickle.loads(
        #     open('param_mountain.npy', 'rb').read())
        param = np.zeros((100))
        #
        # env = GridWorld(randomized_initial=False, direction='border', fail_prob=0.)
        # _, actions_random, _,rewards_random, states_random = create_batch_trajectories(env,  2, 30, param, 0.1)
        env = ContinuousMountainCar(rew_basis=[15,7], randomized_start=True)
        # env = GridWorld(rew_weights=np.random.random(16))
        _, actions_random, _,rewards_random, states_random = create_batch_trajectories(env, 100, 300, param, 0.1)
        # np.random.shuffle(rewards)
        re = RelativeEntropyIRL(reward_features=rewards[:j], reward_random=rewards_random, trajectories_expert=states[:j],
                                trajectories_random=states_random[:j], gamma=.99, horizon=30, n_states=81,  n_actions=4)
        weights[i] = re.fit()

    return weights


if __name__ == '__main__':
    seeds = [np.random.randint(1000000) for _ in range(20)]
    results = Parallel(n_jobs=10, backend='loky')(
        delayed(run)(id, seed) for id, seed in zip(range(2,12), seeds))
    np.save('weights_RE_mountain.npy', results)

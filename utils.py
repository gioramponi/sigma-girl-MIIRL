import numpy as np
from gym.spaces import Box, Discrete
import tensorflow as tf
from baselines.common.policies_bc import build_policy
import baselines.common.tf_util as U
from baselines.common.models import mlp
import pickle
from policies.linear_gaussian_policy import LinearGaussianPolicy


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2, degrees=False):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if degrees:
        return np.degrees(rad)
    return rad


def check_minimum(pi, behavioral_pi, omega, trajectories, episode_length=50, use_baseline=False,
                  use_mask=False, scale_features=1, features_idx=[0, 1, 2]):
    weights = pi.get_weights()
    new_weights = behavioral_pi.get_weights()
    base_dir = weights - new_weights

    print("Norm base dir:", np.linalg.norm(base_dir))
    # pi.set_weights(new_weights)
    states, actions, features, dones = trajectories
    grad, inf = compute_gradient(pi, states, actions, features, dones,
                                 episode_length=episode_length,
                                 omega=omega,
                                 use_baseline=use_baseline,
                                 use_mask=use_mask,
                                 scale_features=scale_features,
                                 features_idx=features_idx,
                                 behavioral_pi=behavioral_pi)
    print("Grad dir:", np.linalg.norm(grad))
    angle = angle_between(base_dir, grad.mean(axis=0), degrees=True)
    # pi.set_weights(weights)
    print(angle)
    input()
    return angle


def compute_gradient(pi, x_dataset, y_dataset, r_dataset, dones_dataset,
                     episode_length, discount_f=0.9999, features_idx=[0, 1, 2],
                     omega=None, normalize_f=False, verbose=False,
                     use_baseline=False, use_mask=False, scale_features=1, behavioral_pi=None, filter_gradients=False,
                     seed=None):
    steps = len(x_dataset)
    r_dim = len(features_idx)
    logger = {}
    # discount factor vector over a finite episode
    gamma = []
    for t in range(episode_length):
        gamma.append(discount_f ** t)
    if seed is not None:
        np.random.seed(seed)
    # discounted reward features computation
    if dones_dataset:
        num_episodes = np.sum(dones_dataset)
    else:
        num_episodes = int(steps // episode_length)
    if verbose:
        print("Episodes:", num_episodes)

    discounted_phi = []
    for episode in range(num_episodes):
        base = episode * episode_length
        r = np.array(r_dataset[base: base + episode_length]).T
        # b = np.random.binomial(1, 1, size=EPISODE_LENGTH)
        reward = []
        try:
            for idx in features_idx:
                reward.append(r[idx])
        except:
            print("Episode:", episode)
            raise ValueError("Dataset corrupted")
        reward = np.array(reward).T * scale_features
        if omega is not None:
            assert len(omega) == len(features_idx), "Features and weights different dimensionality"
            reward = (reward * omega).sum(axis=-1)
            discounted_phi.append(reward * gamma)
        else:
            discounted_phi.append(reward * np.tile(gamma, (r_dim, 1)).transpose())
    discounted_phi = np.array(discounted_phi)
    expected_discounted_phi = discounted_phi.sum(axis=0).sum(axis=0) / num_episodes
    print('Featrues Expectations:', expected_discounted_phi)
    # normalization factor computation
    if normalize_f:
        discounted_phi = np.array(discounted_phi)
        expected_discounted_phi = discounted_phi.sum(axis=0).sum(axis=0) / num_episodes
        if verbose:
            print('Expected discounted phi = ', expected_discounted_phi)
            input()
        # print('Expected discounted phi = ', expected_discounted_phi)
        # input()
        logger['expected_discounted_phi'] = expected_discounted_phi
        expected_discounted_phi = np.tile(expected_discounted_phi, (num_episodes, episode_length, 1))
        discounted_phi /= expected_discounted_phi

    # computing the gradients of the logarithm of the policy wrt policy parameters
    episode_gradients = []
    probs = []

    for step in range(steps):

        step_layers = pi.compute_gradients([x_dataset[step]], [y_dataset[step]])
        step_gradients = []
        for layer in step_layers:
            step_gradients.append(layer.ravel())
        step_gradients = np.concatenate(step_gradients)

        episode_gradients.append(step_gradients)
        if behavioral_pi is not None:
            target_pi_prob = pi.prob(x_dataset[step], y_dataset[step])
            behavioral_pi_prob = behavioral_pi.prob(x_dataset[step], y_dataset[step])
            probs.append(target_pi_prob / (behavioral_pi_prob + 1e-10))

            # print("Step:",step)
    gradients = []
    ratios = []

    for episode in range(num_episodes):
        base = episode * episode_length
        gradients.append(episode_gradients[base: base + episode_length])
        if behavioral_pi is not None:
            ratios.append(probs[base: base + episode_length])

    gradients = np.array(gradients)

    if behavioral_pi is not None:
        ratios = np.array(ratios)
    # GPOMDP optimal baseline computation
    num_params = gradients.shape[2]
    logger['num_params'] = num_params
    if omega is None:
        cum_gradients = np.transpose(np.tile(gradients, (r_dim, 1, 1, 1)), axes=[1, 2, 3, 0]).cumsum(axis=1)
        if behavioral_pi is not None:
            importance_weight = ratios.cumprod(axis=1)
            cum_gradients = cum_gradients * importance_weight
        phi = np.transpose(np.tile(discounted_phi, (num_params, 1, 1, 1)), axes=[1, 2, 0, 3])

    else:
        cum_gradients = gradients.cumsum(axis=1)
        if behavioral_pi is not None:
            importance_weight = ratios.cumprod(axis=1)
            cum_gradients = cum_gradients * importance_weight
        phi = np.transpose(np.tile(discounted_phi, (num_params, 1, 1)), axes=[1, 2, 0])

    '''
    # Freeing memory
    del X_dataset
    del y_dataset
    del r_dataset
    del episode_gradients
    del gamma
    del discounted_phi
    '''
    # GPOMDP objective function gradient estimation
    if use_baseline:
        num = (cum_gradients ** 2 * phi).sum(axis=0)
        den = (cum_gradients ** 2).sum(axis=0) + 1e-10
        baseline = num / den
        if omega is None:
            baseline = np.tile(baseline, (num_episodes, 1, 1, 1))
        else:
            baseline = np.tile(baseline, (num_episodes, 1, 1))
        if use_mask and dones_dataset is not None:
            mask = np.array(dones_dataset).reshape((num_episodes, episode_length))
            for ep in range(num_episodes):
                for step in range(episode_length):
                    if mask[ep, step] == 1.:
                        break
                    mask[ep, step] = 1
            if omega is None:
                baseline *= np.tile(mask, (num_params, r_dim, 1, 1)).transpose((2, 3, 0, 1))
            else:
                baseline *= np.tile(mask, (num_params, 1, 1)).transpose((1, 2, 0))

        phi = phi - baseline

    estimated_gradients = (cum_gradients * (phi)).sum(axis=1)

    if filter_gradients:
        estimated_gradients = filter_grads(estimated_gradients, verbose=verbose)
    return estimated_gradients, {'logger': logger}


def compute_gradient_imp(pi, x_dataset, y_dataset, r_dataset, dones_dataset,
                         episode_length, discount_f=0.9999, features_idx=[0, 1, 2],
                         omega=None, normalize_f=False, verbose=False,
                         use_baseline=False, use_mask=False, scale_features=1, behavioral_pi=None,
                         filter_gradients=False,
                         seed=None):
    steps = len(x_dataset)
    r_dim = len(features_idx)
    logger = {}
    # discount factor vector over a finite episode
    gamma = []
    for t in range(episode_length):
        gamma.append(discount_f ** t)
    if seed is not None:
        np.random.seed(seed)
    # discounted reward features computation
    num_episodes = np.sum(dones_dataset)
    if verbose:
        print("Episodes:", num_episodes)

    discounted_phi = []
    for episode in range(num_episodes):
        base = episode * episode_length
        r = np.array(r_dataset[base: base + episode_length]).T
        # b = np.random.binomial(1, 1, size=EPISODE_LENGTH)
        reward = []
        try:
            for idx in features_idx:
                reward.append(r[idx])
        except:
            print("Episode:", episode)
            raise ValueError("Dataset corrupted")
        reward = np.array(reward).T * scale_features
        if omega is not None:
            assert len(omega) == len(features_idx), "Features and weights different dimensionality"
            reward = (reward * omega).sum(axis=-1)
            discounted_phi.append(reward * gamma)
        else:
            discounted_phi.append(reward * np.tile(gamma, (r_dim, 1)).transpose())
    discounted_phi = np.array(discounted_phi)
    expected_discounted_phi = discounted_phi.sum(axis=0).sum(axis=0) / num_episodes
    print('Featrues Expectations:', expected_discounted_phi)
    # normalization factor computation
    if normalize_f:
        discounted_phi = np.array(discounted_phi)
        expected_discounted_phi = discounted_phi.sum(axis=0).sum(axis=0) / num_episodes
        if verbose:
            print('Expected discounted phi = ', expected_discounted_phi)
            input()
        # print('Expected discounted phi = ', expected_discounted_phi)
        # input()
        logger['expected_discounted_phi'] = expected_discounted_phi
        expected_discounted_phi = np.tile(expected_discounted_phi, (num_episodes, episode_length, 1))
        discounted_phi /= expected_discounted_phi

    # computing the gradients of the logarithm of the policy wrt policy parameters
    episode_gradients = []
    probs = []

    for step in range(steps):

        step_layers = pi.compute_gradients_imp([x_dataset[step]], [y_dataset[step]])
        step_gradients = []
        for layer in step_layers:
            step_gradients.append(layer.ravel())
        step_gradients = np.concatenate(step_gradients)

        episode_gradients.append(step_gradients)
        if behavioral_pi is not None:
            target_pi_prob = pi.prob(x_dataset[step], y_dataset[step])
            behavioral_pi_prob = behavioral_pi.prob(x_dataset[step], y_dataset[step])
            probs.append(target_pi_prob / (behavioral_pi_prob + 1e-10))

            # print("Step:",step)
    gradients = []
    ratios = []

    for episode in range(num_episodes):
        base = episode * episode_length
        gradients.append(episode_gradients[base: base + episode_length])
        if behavioral_pi is not None:
            ratios.append(probs[base: base + episode_length])

    gradients = np.array(gradients)

    if behavioral_pi is not None:
        ratios = np.array(ratios)
    # GPOMDP optimal baseline computation
    num_params = gradients.shape[2]
    logger['num_params'] = num_params
    if omega is None:
        cum_gradients = np.transpose(np.tile(gradients, (r_dim, 1, 1, 1)), axes=[1, 2, 3, 0]).cumsum(axis=1)
        if behavioral_pi is not None:
            importance_weight = ratios.cumprod(axis=1)
            cum_gradients = cum_gradients * importance_weight
        phi = np.transpose(np.tile(discounted_phi, (num_params, 1, 1, 1)), axes=[1, 2, 0, 3])

    else:
        cum_gradients = gradients.cumsum(axis=1)
        if behavioral_pi is not None:
            importance_weight = ratios.cumprod(axis=1)
            cum_gradients = cum_gradients * importance_weight
        phi = np.transpose(np.tile(discounted_phi, (num_params, 1, 1)), axes=[1, 2, 0])

    '''
    # Freeing memory
    del X_dataset
    del y_dataset
    del r_dataset
    del episode_gradients
    del gamma
    del discounted_phi
    '''
    # GPOMDP objective function gradient estimation
    if use_baseline:
        num = (cum_gradients ** 2 * phi).sum(axis=0)
        den = (cum_gradients ** 2).sum(axis=0) + 1e-10
        baseline = num / den
        if omega is None:
            baseline = np.tile(baseline, (num_episodes, 1, 1, 1))
        else:
            baseline = np.tile(baseline, (num_episodes, 1, 1))
        if use_mask:
            mask = np.array(dones_dataset).reshape((num_episodes, episode_length))
            for ep in range(num_episodes):
                for step in range(episode_length):
                    if mask[ep, step] == 1.:
                        break
                    mask[ep, step] = 1
            if omega is None:
                baseline *= np.tile(mask, (num_params, r_dim, 1, 1)).transpose((2, 3, 0, 1))
            else:
                baseline *= np.tile(mask, (num_params, 1, 1)).transpose((1, 2, 0))

        phi = phi - baseline

    estimated_gradients = (cum_gradients * (phi)).sum(axis=1)

    if filter_gradients:
        estimated_gradients = filter_grads(estimated_gradients, verbose=verbose)
    return estimated_gradients, {'logger': logger}


def filter_grads(estimated_gradients, verbose=False):
    filter = None
    num_episodes, num_parameters, num_objectives = estimated_gradients.shape[:]
    mean = estimated_gradients.mean(axis=0)
    std = estimated_gradients.std(axis=0)
    for i in range(num_objectives):
        indexes_std = np.argwhere(np.isclose(std[:, i], 0)).ravel()
        indexes_mean = np.argwhere(np.isclose(mean[:, i], 0)).ravel()
        indeces = np.intersect1d(indexes_std, indexes_mean)
        if len(indeces) > 0:
            if filter is None:
                filter = indeces
            else:
                filter = np.intersect1d(filter, indeces)
    if verbose:
        print("Filtered Indeces:", filter)
    if filter is not None and len(filter) > 0:
        estimated_gradients = np.delete(estimated_gradients, filter, axis=1)

    return estimated_gradients


def load_policy(X_dim, model, num_actions=4, continuous=False, n_bases=50,
                beta=1., trainable_variance=False, init_logstd=-0.4, linear=False, num_layers=0, num_hidden=16):
    if linear:
        policy_train = LinearGaussianPolicy()

        with open(model, "rb") as f:
            if model.endswith('.npy'):
                K = np.load(f)
            else:
                K = pickle.load(f)

        policy_train.set_weights(K.T, np.e ** init_logstd)
        return policy_train

    if continuous:
        observation_space = Box(low=-np.inf, high=np.inf, shape=(X_dim,))
        action_space = Box(low=-1 * np.ones(num_actions), high=np.ones(num_actions))
    else:
        observation_space = Box(low=-np.inf, high=np.inf, shape=(X_dim,))
        action_space = Discrete(num_actions)

    tf.reset_default_graph()
    sess = U.make_session(make_default=True)
    network = mlp(num_hidden=num_hidden, num_layers=num_layers)

    if 'checkpoint' in model:
        pi = build_policy(observation_space, action_space, network,
                          train=False, beta=beta,
                          trainable_variance=trainable_variance,
                          init_logstd=init_logstd)
        with tf.variable_scope('pi'):
            policy_train = pi()
        U.initialize()
        policy_train.load(model)

    else:
        try:
            pi = build_policy(observation_space, action_space, network,
                              train=False, beta=beta,
                              trainable_variance=trainable_variance,
                              init_logstd=init_logstd)
            policy_train = pi()
            U.initialize()
            policy_train.load(model)
        except KeyError:
            sess.close()
            tf.reset_default_graph()
            sess = U.make_session(make_default=True)
            network = mlp(num_hidden=num_hidden, num_layers=num_layers)
            pi = build_policy(observation_space, action_space, network,
                              train=False, beta=beta,
                              trainable_variance=trainable_variance,
                              init_logstd=init_logstd)
            with tf.variable_scope('pi'):
                policy_train = pi()
            U.initialize()
            policy_train.load(model)
    return policy_train


def estimate_cov(estimated_gradients, diag=False):
    _, num_episodes = estimated_gradients.shape[:]
    if diag:
        sigma = np.diag(np.std(estimated_gradients, axis=1) ** 2)
    else:
        sigma = np.cov(estimated_gradients)
    n = sigma.shape[0]
    m = np.trace(sigma) / n
    d_sym = sigma - m * np.eye(n)
    d = np.trace(np.dot(d_sym, d_sym.T)) / n
    prod = 0
    for ep in range(num_episodes):
        if n > 1000:
            print(ep)
        column = estimated_gradients[:, ep].reshape((-1, 1))
        prod_sym = np.dot(column, column.T) - sigma
        prod += np.trace(np.dot(prod_sym, prod_sym.T)) / n
    prod /= (num_episodes ** 2)
    b = np.minimum(prod, d)
    a = d - b
    return b / d * m * np.eye(n) + a / d * sigma


def estimate_distribution_params(estimated_gradients, identity=False, diag=False, cov_estimation=False,
                                 other_options=None, girl=False):
    assert ((cov_estimation or diag) != identity or cov_estimation == diag == identity == False)
    num_episodes, num_parameters, num_objectives = estimated_gradients.shape[:]
    est_grad = estimated_gradients.transpose((1, 2, 0)).reshape((num_parameters * num_objectives, num_episodes),
                                                                order='F')
    mu = np.mean(est_grad, axis=1)

    if other_options[0]:
        sigma = np.ones((len(mu), len(mu)))
    elif other_options[1]:
        sigma = np.kron(np.ones((num_objectives, num_objectives)), np.eye(num_parameters))
    elif len(other_options) >= 3 and other_options[2]:
        if cov_estimation:
            sigma = estimate_cov(est_grad, diag)
        elif diag:
            sigma = np.diag(np.std(est_grad, axis=1) ** 2)
        else:
            sigma = np.cov(est_grad)
        sigma = sigma * np.kron(np.ones((num_objectives, num_objectives)), np.eye(num_parameters)) / num_episodes
    elif len(other_options) >= 3 and other_options[2]:
        # block covariance
        if cov_estimation:
            sigma = estimate_cov(estimated_gradients, diag)
        elif diag:
            sigma = np.diag(np.std(estimated_gradients, axis=1) ** 2)
        else:
            sigma = np.cov(estimated_gradients)
        sigma = sigma * np.kron(np.ones((num_objectives, num_objectives)), np.eye(num_parameters)) / num_episodes
    elif identity:
        sigma = np.eye(len(mu))
    elif cov_estimation:
        sigma = estimate_cov(est_grad, diag) / num_episodes
    elif girl:
        sigma = np.kron(np.ones((num_objectives, num_objectives)), np.eye(num_parameters))
    else:
        if diag:
            sigma = np.diag(np.std(est_grad, axis=1) ** 2)
        else:
            sigma = np.cov(est_grad)
        sigma /= num_episodes

    return mu, sigma


# Enumerate partitions of a set
def algorithm_u(ns, m):
    def visit(n, a):
        ps = [[] for i in range(m)]
        for j in range(n):
            ps[a[j + 1]].append(ns[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                yield visit(n, a)
                a[nu] = a[nu] + 1
            yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

    n = len(ns)
    a = [0] * (n + 1)
    for j in range(1, m + 1):
        a[n - m + j] = j - 1
    return f(m, n, 0, n, a)

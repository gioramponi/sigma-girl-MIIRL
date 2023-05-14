import numpy as np
from math import floor
from autograd import grad


##Implementation of our algorithm

def _stateIndex(s, states_dict):
    return states_dict[f"{s[0]}_{s[1]}_{s[2]}_{s[3]}_{s[4]}"]


# ! PROBLEM
def likelihood_sum(states, prefs, features, K, l, W_l, W, z):
    log_likelihood_sum = 0.0
    for (i, j) in prefs:
        for m in range(K):
            # ! idk, notation error maybe --no y?
            log_likelihood_sum += np.log(get_pref_likelihood(i, j, l, m, features, W)) * z[i,j,l,m]
    return log_likelihood_sum


# ! PROBLEM
def maximum_likelihood_irl(states, prefs, K, intention_l, W, z, n_iteration=10, gradient_iterations=100):
    # init reward weights for the features under intention i
    param = np.array(W[intention_l])
    objective_fn = lambda W_l: -likelihood_sum(states, prefs, K, intention_l, W_l, W, z)
    grad_fn = grad(objective_fn)
    # ! how to do this
    for i in range(n_iteration):
        gradients = grad_fn(param)
        param += .1 * gradients     # alpha = 0.1
    return param


def multiple_intention_irl(states, actions, prefs, len_trajs, num_features, K, n_iterations=20, tolerance=1.e-5):
    # define joint prior probabilities, K = num_clusters
    rho_s = np.ones((K, K))
    rho_s /= K**2

    # initialize reward weights for the features for each cluster
    # 5 features: speed, num_collisions (+neg), num_offroad_visits (+neg)
    W = np.random.random((K, num_features + 1)) # +1 for the actions

    # probability of assigning traj i to intention k
    # randomly initialize, shape = (k, num_trajs)
    z = np.random.random((K, len(states)))
    z /= np.sum(z, axis=0)[np.newaxis, :]   ## normalize the random nums? -- get probs
                                            ## divide by ways of assigning traj i to any intention
    z = z.T

    # Initialize previous assignment for convergence checking
    prev_assignment = np.ones(z.shape)

    # EM algorithm
    it = 0    # curr_iter
    max_iteration = 3
    # perform EM until parameters converge
    while it < max_iteration and np.max(np.abs(z - prev_assignment)) > tolerance:
        print('Iteration %d' % it)
        prev_assignment = z
        # E-Step
        z = e_step(states, actions, prefs, len_trajs, W, rho_s).T   # ! why transpose
        # M-Step
        for l in range(K):
            # get new reward params for every intention
            W[l, :] = maximum_likelihood_irl(states=states,
                                            prefs=prefs,
                                            num_clusters=K,
                                            intention=l,
                                            params=W,
                                            z=z,
                                            n_iteration=n_iterations)
        # update the joint priors
        for l in range(K):
            for m in range(K):
                # ! PROBLEM: should be fixed now
                # ! but not all N^2 i,j pairs present in z, only ones in prefs
                rho_s[l, m] = np.sum(z[:, :, l, m]) / len(states)**2
        it += 1

    return W


def calculate_feature_vector(states, actions, i, W, len_trajs):
    feature_psi = np.zeros(W.shape[1])
    for t in range(len_trajs[i]):
        feature_psi += np.append(states[i][t], actions[i][t])
    return feature_psi


def get_pref_likelihood(states, actions, i, j, y_i, y_j, len_trajs, W):
    traj_i_features = calculate_feature_vector(states, actions, i, W, len_trajs)
    traj_j_features = calculate_feature_vector(states, actions, j, W, len_trajs)

    dot_i = np.dot(W[y_i], traj_i_features.T)
    dot_j = np.dot(W[y_j], traj_j_features.T)

    # to ensure we dont exponentiate too large values
    max_dot = np.maximum(np.max(dot_j), np.max(dot_i))
    exp_i = np.exp(dot_i - max_dot)
    exp_j = np.exp(dot_j - max_dot)

    # to compute softmax prob, sum for every feature vector in traj i and j
    numerator = np.sum(exp_j)
    denominator = numerator + np.sum(exp_i)

    return numerator / denominator


def compute_likelihood(states, actions, i, j, l, m, prefs, len_trajs, K, W):
    # Compute the joint probability of the data for the current values of l and m
    joint_prob = 1.0
    for (i_prime, j_prime) in prefs:
        for y_i in range(K):
            for y_j in range(K):
                y_sum = 0.0
                for l_prime in range(K):
                    for m_prime in range(K):
                        indicator = int(y_i == l_prime and y_j == m_prime)
                        # ! is this right?
                        # ! PROBLEM: see if can move outside
                        if (i == i_prime): y_i = l
                        if (j == j_prime): y_j = m
                        y_sum += indicator * get_pref_likelihood(states, actions, i_prime, j_prime, y_i, y_j, len_trajs, W)
        joint_prob *= y_sum     # i > j in every (i, j) in prefs

    return joint_prob


def e_step(states, actions, prefs, len_trajs, W_t, rho_s):
    ## states = (40, 40, 3), rho_s = (K=4,)
    N = states.shape[0]
    K = rho_s.shape[0]

    # shape = (num_trajs, num_trajs, k, k) bc prob of assigning traj i, j to intentions l, m
    zeta = np.ones((N, N, K, K))

    # Compute the values of z_{i,j}^{l,m} for all trajs i, j and intentions l, m
    for i in range(len(states)):
        for j in range(len(states)):
            denominator = sum(compute_likelihood(states, actions, i, j, k, h, prefs, len_trajs, K, W_t) \
                                * rho_s[k,h] for k in range(K) for h in range(K))
            for l in range(K):
                for m in range(K):
                    # compute z_{i,j}^{l,m}
                    numerator = rho_s[l,m] * compute_likelihood(states, actions, i, j, l, m, prefs, len_trajs, K, W_t)
                    # ! PROBLEM: complexity, see if fixed
                    zeta[i,j,l,m] = numerator / denominator
            print("finished pref")

    # ! all ones on first iter ??
    return zeta
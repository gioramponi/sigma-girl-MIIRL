import numpy as np
from autograd import grad, jacobian


##Implementation of our algorithm

# ! still need to verify
def likelihood_sum(states, actions, len_trajs, prefs, K, l, W_l, W, z):
    log_likelihood_sum = 0.0
    for (i, j) in prefs:
        for m in range(K):
            # ! idk, notation error maybe --no y?
            # Compute the log-probability of xi_i < xi_j given the current values of theta_l and theta_m
            log_likelihood_sum += np.log(get_pref_likelihood(states, actions, len_trajs, i, j, W_l, W[m], W)) * z[i,j,l,m]
    return -log_likelihood_sum


# ! confused, need to verify
def maximum_likelihood_irl(states, actions, len_trajs, prefs, K, W, z, n_iteration=10, gradient_iterations=100):
    # ! where is num_gradient iters used
    objective_fn = lambda W_l: likelihood_sum(states, actions, len_trajs, prefs, K, l, W_l._value, W, z)
    grad_obj = grad(objective_fn)
    hess_obj = jacobian(grad_obj)

    # update reward weights for every intention l
    for l in range(K):
        # init reward weights
        theta_l = np.array(W[l])
        # ! is this right?
        for i in range(n_iteration):
            grad_l = grad_obj(theta_l)
            hess_l = hess_obj(theta_l)
            theta_l -= np.linalg.solve(hess_l, grad_l)
            # ! do we have learning rate, e.g. alpha
            # param += .1 * gradients     # alpha = 0.001
        W[l, :] = theta_l
    return W


def multiple_intention_irl(states, actions, prefs, len_trajs, num_features, K, n_iterations=20, tolerance=1.e-5):
    # define joint prior probabilities, K = num_clusters
    rho_s = np.ones((K, K))
    rho_s /= K**2

    # initialize reward weights for the features for each cluster
    # 5 features: speed, num_collisions (+neg), num_offroad_visits (+neg)
    W = np.random.random((K, num_features + 1)) # +1 for the actions

    # shape = (num_trajs, num_trajs, k, k) bc prob of assigning traj i, j to intentions l, m
    z = np.random.random((len(states), len(states), K, K))
    z /= np.sum(z, axis=(-1, -2), keepdims=True) # normalize the random nums to get a prob distrib

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
        z = e_step(states, actions, prefs, len_trajs, W, rho_s)
        # M-Step
        # get new reward params for every intention
        W = maximum_likelihood_irl(states=states,
                                    actions=actions,
                                    len_trajs=len_trajs,
                                    prefs=prefs,
                                    K=K,
                                    W=W,
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


def get_pref_likelihood(states, actions, len_trajs, i, j, theta_yi, theta_yj, W):
    traj_i_features = calculate_feature_vector(states, actions, i, W, len_trajs)
    traj_j_features = calculate_feature_vector(states, actions, j, W, len_trajs)

    dot_i = np.dot(theta_yi, traj_i_features)
    dot_j = np.dot(theta_yj, traj_j_features)

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
                if (i == i_prime): y_i = l
                if (j == j_prime): y_j = m
                y_sum += get_pref_likelihood(states, actions, len_trajs, i_prime,
                                             j_prime, W[y_i], W[y_j], W)

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
                    # ! PROBLEM: complexity
                    zeta[i,j,l,m] = numerator / denominator
            print("finished pref")

    # ! all ones on first iter ??
    return zeta
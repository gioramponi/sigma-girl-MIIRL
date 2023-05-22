import numpy as np
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

import os
# import multiprocessing
from joblib import Parallel, delayed
import contextlib
import joblib
from tqdm import tqdm

# FROM: https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
# Progress bar for parallelized processes
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

##Implementation of our algorithm


def maximum_likelihood_irl(states, actions, len_trajs, prefs, K, W, z, n_iterations=10, gradient_iterations=100, alpha=0.001):
    # ! PROBLEM: W not changing
    # ~ Potential Fix:
    #   PyTorch tracks tensor usage through torch functions
    #   Need to use torch.Tensors when doing the loss, backprop, etc.

    # Copy of numpy ndarray
    W2 = W.copy()
    # Get list of tensors similar to before (torch)
    W2_t = [torch.tensor(W2[l], requires_grad=True) for l in range(K)]
    # Setup optimizer with weights (tensors)
    optimizer = optim.Adam(W2_t, lr=alpha)

    W_check = W.copy()

    # Optimize on each weight vector
    # update reward weights for every intention l
    for l in range(K):
        # init reward weights
        # theta_l = optimizer.param_groups[0]['params'][l]
        # theta_l would be weights for intention l, a.k.a. W2_t[l] because we want the tensor that actually represents those weights
        for i in range(n_iterations):
            optimizer.zero_grad()
            # likelihood = 0.0
            likelihood = torch.zeros(1)
            for (i, j) in prefs:
                for m in range(K):
                    # likelihood += np.log(get_pref_likelihood(states, actions, len_trajs, i, j, theta_l.data.numpy(), W[m], W)) * z[i,j,l,m]
                    # Using torch function (.log) and some of the inputs are not tensors (W2_t[l] and W2_t[m])
                    # Altered the get_pref_likelihood function to handle tensors or ndarrays...
                    likelihood += torch.log(get_pref_likelihood(states, actions, len_trajs, i, j, W2_t[l], W2_t[m], W)) * z[i,j,l,m]
            # loss = torch.tensor([-likelihood], requires_grad=True)
            # 'likelihood' tensor should contain the history of "how it got here", 
            #   so we can backprop on it since it can calculate its gradient
            likelihood *= -1
            likelihood.backward()
            optimizer.step()
            # theta_l = optimizer.param_groups[0]['params'][l].detach()
        # Remaking the original ndarray version of the weights from the backpropped tensors
        theta_l = W2_t[l].detach()
        W[l] = theta_l.numpy()

    np.set_printoptions(linewidth=120, formatter=dict(float=lambda x: f"{x:.3f}"))
    print(f"-------------OLD WEIGHTS-------------\n{W_check}")
    print(f"-------------NEW WEIGHTS-------------\n{W}")
    print(f"Updated?: {not np.array_equal(W, W_check)}")

    return W


def print_accuracy(num_trajs, K, z, gt_intents, rho_s, theta):
    intent_map = {'Safe': 0, 'Student': 1, 'Demolition': 2, 'Nasty': 3}
    y_true = np.array([intent_map[intent] for intent in gt_intents])
    intent_pairs_shape = (K, K)

    assigned_intents = np.full((num_trajs), -1)
    for i in range(num_trajs // 2):
        for j in range(num_trajs // 2, num_trajs):
            (l, m) = np.unravel_index(np.argmax(z[i,j]), intent_pairs_shape)
            assigned_intents[i] = l
            assigned_intents[j] = m

    print(f"-------------ASSIGNMENTS-------------")
    for i in range(K):
        cluster_indices = np.where(y_true == i)[0]
        print(f"Intention {i}: {assigned_intents[cluster_indices]}")
    print(f"-------------PRIORS-------------\n{rho_s}")
    print(f"-------------THETAS-------------\n{theta}")


def multiple_intention_irl(states, actions, prefs, len_trajs, num_features, K, gt_intents, n_iterations=20, tolerance=1.e-5):
    # define joint prior probabilities, K = num_clusters
    rho_s = np.ones((K, K))
    rho_s /= K**2

    # initialize reward weights for the features for each cluster
    # 5 features: speed, num_collisions (+neg), num_offroad_visits (+neg)
    W = np.random.random((K, num_features + 1)) # +1 for the actions

    num_trajs = len(states)
    # shape = (num_trajs, num_trajs, k, k) bc prob of assigning traj i, j to intentions l, m
    z = np.random.random((num_trajs, num_trajs, K, K))
    z /= np.sum(z, axis=(-1, -2), keepdims=True) # normalize the random nums to get a prob distrib
    # print(np.sum(z[0,0,:,:]))
    # Initialize previous assignment for convergence checking
    prev_assignment = np.ones(z.shape)

    # EM algorithm
    it = 0    # curr_iter
    max_iteration = 20
    # perform EM until parameters converge
    while it < max_iteration and np.max(np.abs(z - prev_assignment)) > tolerance:
        print('Iteration {0}, convergence {1}'.format(it, np.max(np.abs(z - prev_assignment))))
        prev_assignment = z
        # E-Step
        # ? is zeta 4-way symmetric?
        z = e_step(states, actions, prefs, len_trajs, W, rho_s)
        if not (it % 5):
            print_accuracy(num_trajs, K, z, gt_intents, rho_s, W)
        # M-Step
        # get new reward params for every intention
        W = maximum_likelihood_irl(states=states,
                                    actions=actions,
                                    len_trajs=len_trajs,
                                    prefs=prefs,
                                    K=K,
                                    W=W,
                                    z=z,
                                    n_iterations=n_iterations)
        # update the joint priors
        for l in range(K):
            for m in range(K):
                # ! PROBLEM: should be fixed now
                # ! but not all N^2 i,j pairs present in z, only ones in prefs
                rho_s[l, m] = np.sum(z[:, :, l, m]) / len(states)**2
        it += 1

        # Stopping criterion:
        # np.max(np.abs(z - prev_assignment))

    return W


def calculate_feature_vector(states, actions, i, W, len_trajs):
    feature_psi = np.zeros(W.shape[1])
    for t in range(len_trajs[i]):
        feature_psi += np.append(states[i][t], actions[i][t])
    return feature_psi


def get_pref_likelihood(states, actions, len_trajs, i, j, theta_yi, theta_yj, W):
    # states: ndarray
    # actions: ndarray
    # W: ndarray
    # theta_yi: ndarray OR torch.Tensor
    # theta_yj: ndarray OR torch.Tensor

    traj_i_features = calculate_feature_vector(states, actions, i, W, len_trajs)
    traj_j_features = calculate_feature_vector(states, actions, j, W, len_trajs)

    if isinstance(theta_yi, torch.Tensor) or isinstance(theta_yj, torch.Tensor):
        dot_i = torch.dot(theta_yi, torch.tensor(traj_i_features))
        dot_j = torch.dot(theta_yj, torch.tensor(traj_j_features))
        # to ensure we dont exponentiate too large values
        max_dot = torch.maximum(torch.max(dot_j), torch.max(dot_i))
        exp_i = torch.exp(dot_i - max_dot)
        exp_j = torch.exp(dot_j - max_dot)
        # to compute softmax prob, sum for every feature vector in traj i and j
        numerator = torch.sum(exp_j)
        denominator = numerator + torch.sum(exp_i)
    else:
        dot_i = np.dot(theta_yi, traj_i_features)
        dot_j = np.dot(theta_yj, traj_j_features)
        # to ensure we dont exponentiate too large values
        max_dot = np.maximum(np.max(dot_j), np.max(dot_i))
        exp_i = np.exp(dot_i - max_dot)
        exp_j = np.exp(dot_j - max_dot)
        # to compute softmax prob, sum for every feature vector in traj i and j
        numerator = np.sum(exp_j)
        denominator = numerator + np.sum(exp_i)
    # ! PROBLEM: check for div by zero/nans
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

# Pulled part of the E-Step computation to another function
# just to use some multiprocessing.
# It's nice when available :)
def compute_pref_e(states, actions, i, j, prefs, len_trajs, K, W_t, rho_s):
    denominator = sum(compute_likelihood(states, actions, i, j, k, h, prefs, len_trajs, K, W_t) \
                                * rho_s[k,h] for k in range(K) for h in range(K))
    zeta = np.zeros((K,K))
    for l in range(K):
        for m in range(K):
            # compute z_{i,j}^{l,m}
            numerator = rho_s[l,m] * compute_likelihood(states, actions, i, j, l, m, prefs, len_trajs, K, W_t)
            # ! PROBLEM: complexity
            # zeta[i,j,l,m] = numerator / denominator
            zeta[l,m] = numerator / denominator
    # print(f"finished pref {i},{j}")
    return zeta

def e_step(states, actions, prefs, len_trajs, W_t, rho_s):
    ## states = (40, 40, 3), rho_s = (K=4,)
    N = states.shape[0]
    K = rho_s.shape[0]

    # shape = (num_trajs, num_trajs, k, k) bc prob of assigning traj i, j to intentions l, m
    zeta = np.ones((N, N, K, K))

    # Compute the values of z_{i,j}^{l,m} for all trajs i, j and intentions l, m

    # Build inputs for multiprocessing
    print("Building inputs.")
    inputs = []
    for i in range(len(states)):
        for j in range(len(states)):
            inputs.append((states, actions, i, j, prefs, len_trajs, K, W_t, rho_s))
    # Instead of running inner loops sequentially, run multiple of then at the same time
    # Each one returns the 2D (K x K) probability distribution for trajectory pair i, j
    # Here we just compose them all back into zeta at the end
    with tqdm_joblib(tqdm(desc="Running E-step.", total=len(states)**2)) as progress_bar:
        # Can change the second number depending on how many cpu cores you have access to.
        n_procs = max(1, os.cpu_count()-2)
        r = Parallel(n_jobs=n_procs)(
            delayed(compute_pref_e)(s,a,i,j,p,lt,k,wt,rs) for s,a,i,j,p,lt,k,wt,rs in inputs
        )
    print("Finalizing E-step.")
    count = 0
    for i in range(len(states)):
        for j in range(len(states)):
            # Results should be in same order that they were passed into multiprocessing,
            #   so we can just place them using the same order that we made the inputs
            # https://stackoverflow.com/questions/56659294/does-joblib-parallel-keep-the-original-order-of-data-passed
            # https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
            zeta[i,j,:,:] = r[count]
            count += 1

            # denominator = sum(compute_likelihood(states, actions, i, j, k, h, prefs, len_trajs, K, W_t) \
            #                     * rho_s[k,h] for k in range(K) for h in range(K))
            # for l in range(K):
            #     for m in range(K):
            #         # compute z_{i,j}^{l,m}
            #         numerator = rho_s[l,m] * compute_likelihood(states, actions, i, j, l, m, prefs, len_trajs, K, W_t)
            #         # ! PROBLEM: complexity
            #         zeta[i,j,l,m] = numerator / denominator
            # print("finished pref")

    # ! some ones on first iter ??
    return zeta
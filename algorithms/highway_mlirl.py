import numpy as np
from math import floor


##Implementation of Monica Babes-Vroman Apprenticeship Learning About Multiple Intentions

def _stateIndex(s, states_dict):
    assert(not np.isnan(s[0]))
    s = s.astype(int)
    return states_dict[f"{s[0]}_{s[1]}_{s[2]}_{s[3]}_{s[4]}"]


def get_transition_matrix(states, len_trajs, states_idx_dict, actions, state_space, action_space):
    transition_mat = np.zeros((state_space, action_space, state_space))

    for t in range(len(states)):    # trajs
        for i in range(len_trajs[t] - 1):  # trans states in traj
            curr_state = states[t][i]
            next_state = states[t][i + 1]
            action = int(actions[t][i])

            curr_idx = _stateIndex(curr_state, states_idx_dict)
            next_idx = _stateIndex(next_state, states_idx_dict)

            # Update the transition probability
            transition_mat[curr_idx, action, next_idx] += 1

    # Normalize the transition probabilities
    transition_mat /= np.expand_dims(transition_mat.sum(-1), axis=-1)
    # equal probabilities to all next states if (s, a) not visited in data
    transition_mat = np.nan_to_num(transition_mat, nan=-1000)

    # ? their trans_mat has prob[n,:].sum() = # actions
    # ? may have issue with -1000 (unvisited states)
    # ? for them, all states are visited/can calulate trans_prob
    return transition_mat


def initialize_variables(rewards, param, state_space, action_space):
    q = np.zeros((state_space, action_space))
    V = np.zeros(state_space)
    d_V = np.zeros((state_space, rewards.shape[1]))
    for state in range(state_space):
        for action in range(action_space):
            q[state, action] = np.dot(rewards[state], param)
        V[state] = np.dot(rewards[state], param)
        d_V[state] = rewards[state]
    return q, V, d_V


def print_accuracy(z, gt_intents, rho_s, theta):
    intent_map = {'Safe': 0, 'Student': 1, 'Demolition': 2, 'Nasty': 3}
    y_true = np.array([intent_map[intent] for intent in gt_intents])
    tmp = np.zeros((40, 5))
    tmp[:,0:4] = z
    tmp[:,-1] = y_true
    for i in range(4):
        print("Intention {0} : {1}".format(i, tmp[tmp[:,-1]==i][:,0:4].argmax(axis=-1)))
    print("Priors : ", rho_s)
    print("Thetas : ", theta)


def evaluate_gradients(states, actions, param, len_trajs, states_idx_dict, state_space, action_space, gamma, features, prob, beta, weights, p=0,
                       n_iterations=1):
    num_feat_rewards = len(param)
    q, V, d_V = initialize_variables(features, param, state_space, action_space)
    d_q = np.zeros((state_space, action_space, num_feat_rewards))
    pi = np.zeros((state_space, action_space))
    d_pi = np.zeros((state_space, action_space, num_feat_rewards))
    grad = np.zeros((len(states), len(states[0]), num_feat_rewards))

    for i in range(n_iterations):
        for state in range(state_space):  ##OK
            for action in range(action_space):
                q[state, action] = np.dot(param, features[state]) + gamma * np.dot(prob[state, action, :], V)
                d_q[state, action] = features[state] + gamma * np.dot(prob[state, action, :], d_V)

        exp_q = np.exp(beta * (q - np.max(q, axis=1)[:, np.newaxis]))
        zeta = np.sum(exp_q, axis=1)[:, np.newaxis]
        d_zeta = beta * np.sum(exp_q[:, :, np.newaxis] * d_q, axis=1)

        pi = exp_q / zeta
        d_pi_first = beta * (zeta * exp_q)[:, :, np.newaxis] * d_q
        d_pi_second = exp_q[:, :, np.newaxis] * d_zeta[:, np.newaxis, :]
        d_pi = (d_pi_first - d_pi_second) / zeta[:, :, np.newaxis] ** 2

        V = np.sum(pi * q, axis=1)
        d_V = np.sum(q[:, :, np.newaxis] * d_pi + pi[:, :, np.newaxis] * d_q, axis=1)

    L = 0
    for n in range(len(states)):
        traj_norm_L = 0
        for t in range(len_trajs[n]):
            
            state = _stateIndex(states[n, t], states_idx_dict)
            action = int(actions[n, t])
            grad[n, t] = 1 / (pi[state, action] + 1.e-10) * d_pi[state, action]
            traj_norm_L += beta * (q[state, action] - np.max(q[state], axis=-1)) - np.log(zeta[state])
        grad[n, :] *= weights[n]
        traj_norm_L /= len_trajs[n]
        L += traj_norm_L

    return L, np.mean(np.sum(grad, axis=1), axis=0), q


def maximum_likelihood_irl(states, actions, features, probs, init_param, len_trajs, states_idx_dict, state_space, action_space, beta,
                           q, weights=None, gamma=0.99, n_iteration=10, gradient_iterations=100):
    param = np.array(init_param)
    grad = []
    for i in range(n_iteration):
        L, gradients, q = evaluate_gradients(states, actions, param, len_trajs, states_idx_dict, state_space, action_space, gamma, features,
                                          probs, beta, weights,
                                          n_iterations=gradient_iterations)
        gradients = np.clip(gradients, -1, 1)
        # print(gradients)
        param += .1 * gradients
    
    return param, q, grad


def get_ideal_theta(K, features):
    theta = np.ones((K, features.shape[1]))
    theta[0] = [-1, -1, 0, 1, 1] # Safe
    theta[1] = [-1, 0,  0, 1, 0] # Student
    theta[2] = [1,  0,  0, -1,0] # Demolition
    theta[3] = [1, 1, 0, -1, -1] # Nasty

    # theta[1] = [-1000,  1000,  0, 1000,   -1000] # Student
    # theta[2] = [1000,  -1000,  0, -1000, 1000] # Demolition
    return theta


def multiple_intention_irl(states, actions, features, K, gt_intents, len_trajs, states_idx_dict, state_space, action_space, beta,
                           gamma=0.99, n_iterations=20,
                           tolerance=1.e-5,
                           p=0.):
    ## transition probs
    probs = get_transition_matrix(states, len_trajs, states_idx_dict, actions, state_space, action_space)
    rho_s = np.ones(K)  ## define prior probability
    rho_s = rho_s / np.sum(rho_s)
    theta = np.random.random((K, features.shape[1]))  # reward feature
    theta = get_ideal_theta(K, features)
    q = np.zeros((K, state_space, action_space))  # Q values

    # ? useless stuff, fix
    z = e_step(states, actions, len_trajs, states_idx_dict, theta, rho_s, action_space, q, beta).T  # K, N
    z = np.random.random(z.shape)
    z /= np.sum(z, axis=0)[np.newaxis, :]
    z = z
    prev_assignment = np.ones(z.shape)
    it = 0

    max_iteration = 40
    while it < max_iteration and np.max(np.abs(z - prev_assignment)) > tolerance:
        print('Iteration {0}, convergence {1}'.format(it, np.max(np.abs(z - prev_assignment))))
        prev_assignment = z
        z = e_step(states, actions, len_trajs, states_idx_dict, theta, rho_s, action_space, q, beta).T  # K, N
        if it % 1 == 0:
            print_accuracy(z.T, gt_intents, rho_s, theta)
        it += 1
        for i in range(z.shape[0]):
            theta[i, :], q[i, :], grad = maximum_likelihood_irl(states=states,
                                                                actions=actions,
                                                                features=features,
                                                                probs=probs,
                                                                init_param=theta[i],
                                                                len_trajs=len_trajs,
                                                                states_idx_dict=states_idx_dict,
                                                                state_space=state_space,
                                                                action_space=action_space,
                                                                beta=beta,
                                                                q=q[i],
                                                                weights=z[i],
                                                                gamma=gamma,
                                                                n_iteration=n_iterations)
        rho_s = np.sum(z, axis=1) / len(states)
        # rho_s = np.array([0.25, 0.25, 0.25, 0.25]) # todo: change back, testing smth
    return z, theta

def get_hardcoded_policy(states_idx_dict, action_space, theta):
    pi = np.zeros((len(states_idx_dict), action_space, theta.shape[0]))
    for state, i in states_idx_dict.items():
        stay, left, right,  = 0, 1, -1
        Safe, Student, Demolition, Nasty = 0, 1, 2, 3

        # Demolition: collisions_only
        pi[i][stay][Demolition] = 1 / 3
        pi[i][left][Demolition] = 1 / 3
        pi[i][right][Demolition] = 1 / 3

        # Student: offroad_only
        pi[i][stay][Student] = 0.01 # shouldnt have 0, or will be all 0
        pi[i][left][Student] = 0.495
        pi[i][right][Student] = 0.495

        # Nasty: collisions_and_offroad
        pi[i][stay][Nasty] = 0.2
        pi[i][left][Nasty] = 0.4
        pi[i][right][Nasty] = 0.4

        # Safe: no_collisions_no_offroad
        pi[i][stay][Safe] = 0.9 
        pi[i][left][Safe] = 0.05
        pi[i][right][Safe] = 0.05

    return pi


def e_step(trajs, actions, len_trajs, states_idx_dict, theta, rho_s, action_space, q, beta):
    # ? with hardcoded pi, seems to be working well
    # ? trajs 11, 20, 22, 37 mis-assigned, but reasonable (check sheets)
    # hardcoded_pi = get_hardcoded_policy(states_idx_dict, action_space, theta)
    zeta = np.ones((trajs.shape[0], rho_s.shape[0]))
    for t in range(len(trajs)):
        for s in range(len_trajs[t]):
            state = int(_stateIndex(trajs[t, s], states_idx_dict))
            action = int(actions[t, s])
            if t == 38:
                pass
            for k in range(theta.shape[0]):
                # pi = hardcoded_pi[state, action, k]
                # ? all q roughly equal for every action in an intent
                prob = [np.exp(beta * (q[k, state, act] - max(q[k, state]))) for act in range(action_space)]
                pi = prob[action] / np.sum(prob)
                zeta[t, k] *= pi
        for k in range(theta.shape[0]):
            zeta[t, k] *= rho_s[k]
        zeta[t, :] /= np.sum(zeta[t, :])
    return zeta


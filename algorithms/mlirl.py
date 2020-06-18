import numpy as np
from math import floor


##Implementation of Monica Babes-Vroman Apprenticeship Learning About Multiple Intentions

def _coupleToInt(x, y=None, K=9):
    if y is None:
        y = x[1]
        x = x[0]
    x = int(x)
    y = int(y)
    st = y + x * K
    return int(st)


def compute_probabilities(s, action, state_space, p):
    probabilities_states = np.zeros(state_space)
    K = np.sqrt(state_space)
    x, y = floor(s / K), s % K
    if action == 0:
        y_new = min(y + 1, K - 1)
        probabilities_states[int(_coupleToInt(x, y_new, K))] += 1 - p + p / 4
    else:
        y_new = min(y + 1, K - 1)
        probabilities_states[int(_coupleToInt(x, y_new, K))] += p / 4
    if action == 1:
        x_new = min(x + 1, K - 1)
        probabilities_states[int(_coupleToInt(x_new, y, K))] += 1 - p + p / 4
    else:
        x_new = min(x + 1, K - 1)
        probabilities_states[int(_coupleToInt(x_new, y, K))] += p / 4
    if action == 2:
        y_new = max(y - 1, 0)
        probabilities_states[int(_coupleToInt(x, y_new, K))] += 1 - p + p / 4
    else:
        y_new = min(y - 1, 0)
        probabilities_states[int(_coupleToInt(x, y_new, K))] += p / 4
    if action == 3:
        x_new = max(x - 1, 0)
        probabilities_states[int(_coupleToInt(x_new, y, K))] += 1 - p + p / 4
    else:
        x_new = min(x - 1, 0)
        probabilities_states[int(_coupleToInt(x_new, y, K))] += p / 4
    return probabilities_states / np.sum(probabilities_states)


def create_probabilities(states_space, action_space, p):
    prob = np.zeros((states_space, action_space, states_space))
    for s in range(states_space):
        for a in range(action_space):
            prob[s, a, :] = compute_probabilities(s, a, states_space, p)
    return prob


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


def evaluate_gradients(states, actions, param, state_space, action_space, gamma, rewards, prob, beta, weights, p=0,
                       n_iterations=1):
    num_feat_rewards = len(param)
    q, V, d_V = initialize_variables(rewards, param, state_space, action_space)
    d_q = np.zeros((state_space, action_space, num_feat_rewards))
    pi = np.zeros((state_space, action_space))
    d_pi = np.zeros((state_space, action_space, num_feat_rewards))
    grad = np.zeros((len(states), len(states[0]), num_feat_rewards))

    for i in range(n_iterations):
        for state in range(state_space):  ##OK
            for action in range(action_space):
                q[state, action] = np.dot(param, rewards[state]) + gamma * np.dot(prob[state, action, :], V)
                d_q[state, action] = rewards[state] + gamma * np.dot(prob[state, action, :], d_V)

        exp_q = np.exp(beta * (q - np.max(q, axis=1)[:, np.newaxis]))
        zeta = np.sum(exp_q, axis=1)[:, np.newaxis]
        d_zeta = beta * np.sum(exp_q[:, :, np.newaxis] * d_q, axis=1)

        pi = exp_q / zeta
        d_pi_first = beta * (zeta * exp_q)[:, :, np.newaxis] * d_q
        d_pi_second = exp_q[:, :, np.newaxis] * d_zeta[:, np.newaxis, :]
        d_pi = (d_pi_first - d_pi_second) / zeta[:, :, np.newaxis] ** 2

        V = np.sum(pi * q, axis=1)
        d_V = np.sum(q[:, :, np.newaxis] * d_pi + pi[:, :, np.newaxis] * d_q, axis=1)

    for n in range(len(states)):
        for t in range(len(states[n])):
            state = int(_coupleToInt(states[n, t]))
            action = int(actions[n, t])
            grad[n, t] = 1 / (pi[state, action] + 1.e-10) * d_pi[state, action]
        grad[n, :] *= weights[n]

    return np.mean(np.sum(grad, axis=1), axis=0), q


def maximum_likelihood_irl(states, actions, rewards, probs, init_param, state_space, action_space, beta,
                           q, weights=None, gamma=0.99, n_iteration=10, gradient_iterations=100):
    param = np.array(init_param)
    grad = []
    for i in range(n_iteration):
        gradients, q = evaluate_gradients(states, actions, param, state_space, action_space, gamma, rewards,
                                          probs, beta, weights,
                                          n_iterations=gradient_iterations)
        param += .1 * gradients
    return param, q, grad


def multiple_intention_irl(states, actions, rewards, K, state_space, action_space, beta,
                           gamma=0.99, n_iterations=20,
                           tolerance=1.e-5,
                           p=0.):
    probs = create_probabilities(state_space, action_space, p)
    rho_s = np.ones(K)  ## define prior probability
    rho_s = rho_s / np.sum(rho_s)
    theta = np.random.random((K, 3))  # reward feature
    q = np.zeros((K, state_space, action_space))  # Q values

    z = e_step(states, actions, theta, rho_s, action_space, q, beta).T  # K, N
    z = np.random.random(z.shape)
    z /= np.sum(z, axis=0)[np.newaxis, :]
    z = z
    prev_assignment = np.ones(z.shape)
    it = 0

    max_iteration = 20
    while it < max_iteration and np.max(np.abs(z - prev_assignment)) > tolerance:
        print('Iteration %d' % it)
        prev_assignment = z
        z = e_step(states, actions, theta, rho_s, action_space, q, beta).T  # K, N
        it += 1
        for i in range(z.shape[0]):
            theta[i, :], q[i, :], grad = maximum_likelihood_irl(states=states,
                                                                actions=actions,
                                                                rewards=rewards,
                                                                probs=probs,
                                                                init_param=theta[i],
                                                                state_space=state_space,
                                                                action_space=action_space,
                                                                beta=beta,
                                                                q=q[i],
                                                                weights=z[i],
                                                                gamma=gamma,
                                                                n_iteration=n_iterations)
        rho_s = np.sum(z, axis=1) / len(states)
    n = int(len(states) / 3)

    P_true = [np.concatenate((np.zeros(n), np.ones(n * 2))),
              np.concatenate((np.ones(n), np.zeros(n * 2)))]  # , np.zeros(2*n), np.zeros(2*n), np.zeros(2*n)]

    P_true2 = [np.concatenate((np.ones(n), np.zeros(n * 2))),
               np.concatenate((np.zeros(n), np.ones(n * 2)))]  #
    P_true3 = [np.concatenate((np.zeros(n * 2), np.ones(n))),
               np.concatenate((np.ones(n * 2), np.zeros(n)))]  #
    P_true4 = [np.concatenate((np.ones(n * 2), np.zeros(n))),
               np.concatenate((np.zeros(n * 2), np.ones(n)))]  #

    res1 = np.sum(P_true * z)
    res2 = np.sum(P_true2 * z)
    res3 = np.sum(P_true3 * z)
    res4 = np.sum(P_true4 * z)

    return np.max([res1, res2, res3, res4])


def e_step(states, actions, theta, rho_s, action_space, q, beta):
    zeta = np.ones((states.shape[0], rho_s.shape[0]))
    for n in range(len(states)):
        for t in range(len(states[n])):
            state = int(_coupleToInt(states[n, t]))
            action = int(actions[n, t])
            for k in range(theta.shape[0]):
                prob = [np.exp(beta * (q[k, state, act] - max(q[k, state]))) for act in range(action_space)]
                pi = prob[action] / np.sum(prob)
                zeta[n, k] *= pi
        for k in range(theta.shape[0]):
            zeta[n, k] *= rho_s[k]
        zeta[n, :] /= np.sum(zeta[n, :])
    return zeta


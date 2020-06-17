import pickle
import numpy as np
from math import floor
from envs.grid_word_AB import GridWorld_AB
from forward_policies import policy_boltzmann as ps
from forward_policies.policy_boltzmann import gpomdp

##Implementation of Monica Babes-Vroman Apprenticeship Learning About Multiple Intentions

def _coupleToInt(x, y, K):
    return y + x * K


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


def create_rewards(state_space, goal):
    rewards = np.zeros((state_space, 3))
    K = 5
    for s in range(state_space):
        x, y = np.floor(s / K), s % K
        if int(x) == goal[0] and int(y) == goal[1] : # goal state
            rewards[s][2] = 1
        elif x >= 1 and x <= K-2 and y >= 1 and y <= K-2:  # slow_region
            rewards[s][1] = -1
        else:
            rewards[s][0] = -1
    return rewards


def initialize_variables(rewards, param, state_space, action_space):
    q = np.zeros((state_space,action_space))
    V = np.zeros(state_space)
    d_V = np.zeros((state_space, rewards.shape[1]))
    for state in range(state_space):
        for action in range(action_space):
            q[state, action] = np.dot(rewards[state], param)
        V[state] = np.dot(rewards[state], param)
        d_V[state] = rewards[state]
    return q, V, d_V


def evaluate_gradients(states, actions,param, state_space, action_space, q, goal, beta, weights, n_iterations=4):
    rewards = create_rewards(state_space, goal)
    prob = create_probabilities(state_space, action_space, 0.)
    gamma = 0.99
    num_feat_rewards = 3
    q, V, d_V = initialize_variables(rewards, param, state_space, action_space)
    d_q = np.zeros((state_space,action_space, num_feat_rewards))
    pi = np.zeros((state_space, action_space))
    d_pi = np.zeros((state_space, action_space, num_feat_rewards))
    grad = np.zeros((len(states), 30,num_feat_rewards))
    zeta = np.zeros((state_space))
    d_zeta = np.zeros((state_space, num_feat_rewards))
    for i in range(n_iterations):
        for state in range(state_space):
            for action in range(action_space):
                q[state,action] = np.dot(param, rewards[state]) + gamma * np.sum([prob[state,action,s_]*V[s_] for s_ in range(state_space)], axis=0)
                d_q[state,action] = rewards[state] + gamma * np.sum([prob[state,action,s_] * d_V[s_] for s_ in range(state_space)], axis=0)

        for state in range(state_space):
            zeta[state] = np.sum(np.exp(beta * (q[state] - np.max(q[state]))))
            d_zeta[state] = beta * np.sum(np.exp(beta*(q[state] - np.max(q[state])))[:, np.newaxis] * d_q[state], axis=0)

        # print(zeta, q, i)
        for state in range(state_space):
            for action in range(action_space):
                pi[state, action] = np.exp(beta*(q[state, action] - np.max(q[state]))) / zeta[state]
                d_pi[state, action] = (beta * np.exp(beta*(q[state,action]-np.max(q[state]))) * zeta[state] * d_q[state,action]  -
                             np.exp(beta*(q[state,action]-np.max(q[state]))) * d_zeta[state]) / (zeta[state] ** 2)

        for state in range(state_space):
            V[state] = np.sum([q[state,a_]*pi[state,a_] for a_ in range(action_space)])
            d_V[state] = np.sum([d_q[state,a_]*pi[state,a_]+q[state,a_]*d_pi[state,a_] for a_ in range(action_space)], axis=0)

    for n in range(len(states)):
        for t in range(len(states[n])):
            state = int(states[n][t])
            action = int(actions[n][t])
            grad[n,t] = 1/(pi[state,action] + 1.e-10)*d_pi[state,action]
        grad[n,:] *= weights[n]

    return np.mean(np.mean(grad, axis=0), axis=0), q


def maximum_likelihood_irl(states, actions, init_param, state_space, action_space, n_iteration, beta, q, weights=None):
    param = np.array(init_param)
    grad = []
    goal = np.array([4., 2.])
    gradients = 1
    for i in range(30):
        gradients, q = evaluate_gradients(states, actions, param, state_space, action_space, q, goal, beta, weights, n_iterations=10)
        param = param + 0.01 * gradients
    return param, q, grad


def multiple_intention_irl(states, actions, K, state_space, action_space, beta):
    rho_s = np.ones(K)
    rho_s = rho_s / np.sum(rho_s)
    theta = np.random.random((K, 3))
    q = np.zeros((K,state_space, action_space))
    for i in range(10):
        z = e_step(states, actions, theta, rho_s, action_space, q, beta).T #K, N
        for i in range(z.shape[0]):
            theta[i, :], q[i, :], grad = maximum_likelihood_irl(states, actions, theta[i], state_space, action_space, 20,
                                                          beta, q[i], z[i])
        # print(z)
        rho_s = np.sum(z, axis=1) / len(states)
    first_cluster = []
    second_cluster = []
    cl = int(len(states)/3)
    for i in range(K):
        first_cluster.append(np.sum(z[i][:cl])+np.sum(z[i][cl*2:]))
        second_cluster.append(np.sum(z[i][cl:cl*2]))
    theta = [theta[np.argmax(first_cluster)], theta[np.argmax(second_cluster)]]
    return theta


def e_step(states, actions, theta, rho_s, action_space, q, beta):
    zeta = np.ones((states.shape[0], rho_s.shape[0]))
    for n in range(len(states)):
        for t in range(len(states[n])):
            state = int(states[n][t])
            action = int(actions[n][t])
            for k in range(theta.shape[0]):
                prob = [np.exp(beta * (q[k, state, act] - max(q[k, state]))) for act in range(action_space)]
                pi = prob[action] / np.sum(prob)
                zeta[n, k] *= pi
        for k in range(theta.shape[0]):
            zeta[n,k] *= rho_s[k]
        zeta[n,:] /= np.sum(zeta[n,:])

    return zeta


if __name__=='__main__':
    env2 = GridWorld_AB(size=5,goal=(4,2), start=(0,2), direction='up',randomized_start=False,fail_prob=0.)#
    env3 = GridWorld_AB(size=5,goal=(4,2), start=(0,0), direction='up',randomized_start=False,fail_prob=0.)#
    param_up = pickle.loads(open('param_discrete_up', 'rb').read())
    param_down = pickle.loads(open('param_discrete_down', 'rb').read())
    param_center = pickle.loads(open('param_discrete_center', 'rb').read())

    state_space = 25
    action_space = 4
    # ps.create_batch_trajectories(env2, 10, 30, param_up, state_space, action_space, True)
    s_center, a_center, _ = ps.create_batch_trajectories(env2, 100, 20, param_center, state_space, action_space, False)
    s_down, a_down, _ = ps.create_batch_trajectories(env2,100, 20, param_down, state_space, action_space, False)
    s_up, a_up,_ =ps.create_batch_trajectories(env2, 100, 20, param_up, state_space, action_space, False)
    rewards = create_rewards(state_space, goal=(4,2))
    s = np.concatenate((s_up, s_center, s_down))
    a = np.concatenate((a_up, a_center, a_down))
    beta = 0.5
    param = multiple_intention_irl(s, a, 5, state_space, action_space, beta)
    print(param)

    env = GridWorld_AB(randomized_start=True, goal=(4,2), start=(0,2), weights=param[0], size=5)

    param,_,_,rewards, gradients = gpomdp(env, 100, 30, 30, np.random.random((state_space * action_space)), 0.99, state_space, action_space)

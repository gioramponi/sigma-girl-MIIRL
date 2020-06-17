import numpy as np


class MDP(object):
    """
    A class representing a Markov decision process (MDP).

    Parameters
    ----------
        - n_states: the number of states (S)
        - n_actions: the number of actions (A)
        - transitions: transition probability matrix P(S_{t+1}=s'|S_t=s,A_t=a). Shape: (AxSxS)
        - rewards: the reward function R(s,a) or R(s,a,s'). Shape: (A,S) or (A,S,S)
        - init_state: the initial state probabilities P(S_0=s). Shape: (S,)
        - gamma: discount factor in [0,1]

    """

    def __init__(self, n_states, n_actions, transitions, rewards, init_state, gamma):

        assert n_states > 0, "The number of states must be positive"
        self.S = n_states
        assert n_actions > 0, "The number of actions must be positive"
        self.A = n_actions

        assert 0 <= gamma < 1, "Gamma must be in [0,1)"
        self.gamma = gamma

        assert transitions.shape == (n_actions, n_states, n_states), "Wrong shape for P"
        self.P = [transitions[a] for a in range(self.A)]

        if rewards.shape == (n_actions, n_states, n_states):
            # Compute the expected one-step rewards
            self.R = np.sum(self.P * rewards, axis=2)
        elif rewards.shape == (n_actions, n_states):
            self.R = rewards
        else:
            raise TypeError("Wrong shape for R")

        assert init_state.shape == (n_states,), "Wrong shape for P0"
        self.P0 = init_state

    def bellmap_op(self, V):
        """
        Applies the optimal Bellman operator to a value function V.

        :param V: a value function. Shape: (S,)
        :return: the updated value function and the corresponding greedy action for each state. Shapes: (S,) and (S,)
        """

        assert V.shape == (self.S,), "V must be an {0}-dimensional vector".format(self.S)

        Q = np.empty((self.A, self.S))
        for a in range(self.A):
            Q[a] = self.R[a] + self.gamma * self.P[a].dot(V)

        return Q.argmax(axis=0), Q.max(axis=0), Q

    def value_iteration(self, max_iter=1000, tol=1e-3, verbose=False,qs =False):
        """
        Applies value iteration to this MDP.

        :param max_iter: maximum number of iterations
        :param tol: tolerance required to converge
        :param verbose: whether to print info
        :return: the optimal policy and the optimal value function. Shapes: (S,) and (S,)
        """

        # Initialize the value function to zero
        V = np.zeros(self.S,)

        for i in range(max_iter):

            # Apply the optimal Bellman operator to V
            pi, V_new, Q = self.bellmap_op(V)

            # Check whether the difference between the new and old values are below the given tolerance
            diff = np.max(np.abs(V - V_new))

            if verbose:
                print("Iter: {0}, ||V_new - V_old||: {1}, ||V_new - V*||: {2}".format(i, diff,
                                                                                      2*diff*self.gamma/(1-self.gamma)))

            # Terminate if the change is below tolerance
            if diff <= tol:
                break

            # Set the new value function
            V = V_new
        if qs:
            return pi, V, Q
        else:
            return pi, V


def random_mdp(n_states, n_actions, gamma=0.99):
    """
    Creates a random MDP.

    :param n_states: number of states
    :param n_actions: number of actions
    :param gamma: discount factor
    :return: and MDP with S state, A actions, and randomly generated transitions and rewards
    """

    # Create a random transition matrix
    P = np.random.rand(n_actions, n_states, n_states)
    # Make sure the probabilities are normalized
    for s in range(n_states):
        for a in range(n_actions):
            P[a, s, :] = P[a, s, :] / np.sum(P[a, s, :])

    # Create a random reward matrix
    R = np.random.rand(n_actions, n_states)

    # Create a random initial-state distribution
    P0 = np.random.rand(n_states)
    # Normalize
    P0 /= np.sum(P0)

    return MDP(n_states, n_actions, P, R, P0, gamma)
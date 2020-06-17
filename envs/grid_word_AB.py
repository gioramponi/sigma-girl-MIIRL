from forward_policies import policy_boltzmann as ps
import numpy as np
import gym
from gym import spaces
# from gym.envs.classic_control import rendering
from builtins import AttributeError
from math import floor
from envs.mdp import MDP
import pickle

class GridWorld(gym.Env):

    ACTION_LABELS = ["UP", "RIGHT", "DOWN", "LEFT"]
    """
    A KxK discrete gridworld environment.
    
    State space: discrete in {0,1,...,K^2-1}
    Action space: discrete in {0,1,2,3}, where 0 is north, 1 is east, 2 is south, and 3 is west
    
    Reward: 1 for reaching the goal, 0 otherwise.
    
    Parameters
    ----------
        - horizon: maximum number of time steps
        - shape: shape of the grid
        - fail_prob: probability of failing an action
        - goal: goal position (x,y)
        - start: start position (x,y)
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, shape=(7, 7), horizon=10, fail_prob=0.1, goal=(4, 2), start=(0, 2), direction ='center',
                 gamma=0.99, rew_weights=[1,20,1], randomized_initial=False, extended_features=False):

        assert shape[0] >= 3 and shape[1] >= 3, "The grid must be at least 3x3"
        self.H = 2 * shape[0] +1 #mirrored grid
        self.W = shape[1]
        assert horizon >= 1, "The horizon must be at least 1"
        self.horizon = horizon
        assert 0 <= fail_prob <= 1, "The probability of failure must be in [0,1]"
        self.fail_prob = fail_prob
        self.direction = direction

        if goal is None:
            goal = (shape[1]-1, shape[0])
        if start is None:
            start = (0, shape[0])
        self.done = False
        self.init_state = self._coupleToInt(start[0], start[1])
        self.randomized_start = randomized_initial
        self.goal_state = self._coupleToInt(goal[0], goal[1])
        self.PrettyTable = None
        self.rendering = None
        self.gamma = gamma
        if rew_weights is None:
            rew_weights = [1, 10, 0]
        self.rew_weights = np.array(rew_weights)
        # gym attributes
        self.viewer = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.W * self.H)
        self.ohe = np.eye(self.W*self.H)
        self.extended_features = extended_features
        mu, p, r = self.calculate_mdp()

        self.mu = mu
        self.p = p
        self.r = r
        self.n_states = self.W * self.H  # Number of states
        self.n_actions = 4  # Number of actions
        # initialize state
        self.reset()

    def _coupleToInt(self, x, y):
        return y + x * self.H

    def _intToCouple(self, n):
        return floor(n / self.H), n % self.H

    def get_rew_features(self, state=None):
        if state is None:
            if self.done:
                return np.zeros(3)
            state = self.state
        x, y = self._intToCouple(state)
        features = np.zeros(3)

        if state == self.goal_state: #goal state
            features[2] = 1
        elif x >= 1 and x <= self.W-2 and y >= 1 and y <= self.H-2:  # slow_region
            features[1] = -np.linalg.norm(self.state - self.goal)
        elif self.direction == 'up' and y < (self.H-1)/2:
            features[1] = -np.linalg.norm(self.state - self.goal)
        elif self.direction == 'down' and y > (self.H-1)/2:
            features[1] = -np.linalg.norm(self.state - self.goal)
        else:
            features[0] = -np.linalg.norm(self.state - self.goal)   # fast region
        return features

    def step(self, action, ohe=False):
        if self.state == self.goal_state:
            return self.get_state(ohe), 0, 1, {'features': np.zeros(3)}
        x, y = self._intToCouple(self.state)
        action = np.random.choice(4) if np.random.rand() < self.fail_prob else action

        if action == 0:
            y = min(y + 1, self.H - 1)
        elif action == 1:
            x = min(x + 1, self.W - 1)
        elif action == 2:
            y = max(y - 1, 0)
        elif action == 3:
            x = max(x - 1, 0)
        else:
            raise AttributeError("Illegal action")

        self.state = self._coupleToInt(x, y)
        features = self.get_rew_features()
        reward = np.sum(self.rew_weights * features)
        self.done = 1 if self.state == self.goal_state else 0
        if self.extended_features:
            features = np.zeros(self.n_states)
            features[self.state] = reward
        return self.get_state(ohe), reward, self.done, {'features': features}

    def get_state(self, ohe=False):
        if ohe:
            return self.ohe[self.state]
        return self.state


    def reset(self, state=None):
        if state is None:
            if self.randomized_start:
                self.state = np.random.choice(self.observation_space.n,p = self.mu)
            else:
                self.state = self.init_state
        else:
            self.state = state
        self.done = False
        return self.get_state(ohe)

    def _render(self, mode='human', close=False):
        if not self.rendering:
            from gym.envs.classic_control import rendering as rend
            self.rendering = rend
        rendering = self.rendering
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(0, self.W, 0, self.H)

        # Draw the grid
        for i in range(self.W):
            self.viewer.draw_line((i, 0), (i, self.H))
        for i in range(self.H):
            self.viewer.draw_line((0, i), (self.W, i))

        goal = self.viewer.draw_circle(radius=0.5)
        goal.set_color(0, 0.8, 0)
        goal_x, goal_y = self._intToCouple(self.goal_state)
        goal.add_attr(rendering.Transform(translation=(goal_x + 0.5, goal_y + 0.5)))

        agent = self.viewer.draw_circle(radius=0.4)
        agent.set_color(.8, 0, 0)
        agent_x, agent_y = self._intToCouple(self.state)
        print(self.state, agent_x, agent_y)
        transform = rendering.Transform(translation=(agent_x + 0.5, agent_y + 0.5))
        agent.add_attr(transform)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def render_pi_v(self, pi, V):
        if self.PrettyTable is None:
            from prettytable import PrettyTable as pt
            self.PrettyTable = pt

        t = self.PrettyTable()
        pi_table = np.zeros((self.H, self.W), dtype=np.int)
        v_table = np.zeros((self.H, self.W))
        for state in range(self.W * self.H):
            x, y = self._intToCouple(state)
            pi_table[self.H - y - 1, x] = int(pi[state])
            v_table[self.H - y - 1, x] = V[state]

        for i in range(self.H):
            row = []
            for j in range(self.W):
                row.append(GridWorld.ACTION_LABELS[pi_table[i, j]] + ":%.2f" % v_table[i, j])
            t.add_row(row)

        print(t)

    def calculate_mdp(self):
        n_states = self.W * self.H  # Number of states
        n_actions = 4  # Number of actions

        # Compute the initiial state distribution
        if self.randomized_initial:
            P0 = np.ones(n_states) * 1 / (n_states - 1)
            P0[self.goal_state] = 0

        else:
            P0 = np.zeros(n_states)
            P0[self.init_state] = 1

        # Compute the reward function
        R = np.zeros((n_actions, n_states, n_states))

        # Compute the transition probability matrix
        P = np.zeros((n_actions, n_states, n_states))
        p = self.fail_prob
        delta_x = [0, 1, 0, -1]  # Change in x for each action [UP, RIGHT, DOWN, LEFT]
        delta_y = [1, 0, -1, 0]  # Change in y for each action [UP, RIGHT, DOWN, LEFT]
        for s in range(n_states):
            rew = np.sum(self.get_rew_features(s) * self.rew_weights)
            R[:, :, s] = rew
            for a in range(n_actions):
                x, y = self._intToCouple(s)  # Get the coordinates of s
                x_new = max(min(x + delta_x[a], self.W - 1), 0)  # Correct next-state for a
                y_new = max(min(y + delta_y[a], self.H - 1), 0)  # Correct next-state for a
                s_new = self._coupleToInt(x_new, y_new)

                P[a, s, s_new] += 1 - p  # a does not fail with prob. 1-p
                # Suppose now a fails and try all other actions
                for a_fail in range(n_actions):
                    x_new = max(min(x + delta_x[a_fail], self.W - 1), 0)  # Correct next-state for a_fail
                    y_new = max(min(y + delta_y[a_fail], self.H - 1), 0)  # Correct next-state for a_fail
                    P[a, s, self._coupleToInt(x_new, y_new)] += p / 4  # a_fail is taken with prob. p/4
        # The goal state is terminal -> only self-loop transitions
        P[:, self.goal_state, :] = 0
        P[:, self.goal_state, self.goal_state] = 1
        R[:, self.goal_state, self.goal_state] = 0  # don't get reward after reaching goal state
        return P0, P, R

    def get_mdp(self):

        """Returns an MDP representing this gridworld"""
        n_states = self.W * self.H  # Number of states
        n_actions = 4  # Number of actions
        P0 = self.mu
        R = self.r
        P = self.p

        return MDP(n_states, n_actions, P, R, P0, self.gamma)

if __name__ == '__main__':
    env = GridWorld_AB(size=5,goal=(4,2), start=(0,2), direction='up', weights=[1,20,1])# coord_x_1=0, coord_x_2=4,coord_y_1=4, coord_y_2=4, center=True)
    state_space = 25
    action_space = 4
    param, results, states, rewards__, gradients__= ps.gpomdp(env, 100, 30, 30, np.random.random((state_space * action_space)), 0.9, state_space, action_space)

    param = pickle.loads(open('param_discrete_center','rb').read())
    param = param.reshape((25,4))
    param_new = np.zeros((25,4))
    for i in range(25):
        m = np.argmax(param[i])
        param_new[i][m] = 3.3
    param = param_new.reshape((-1))
    open('param_discrete_2_center','wb').write(pickle.dumps(param))
    env2 = GridWorld_AB(size=5,goal=(4,2), fail_prob=0.1, start=(0,2), randomized_start=False)#
    # ps.create_batch_trajectories(env2, 10, 20, param, state_space, action_space, True )


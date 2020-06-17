import numpy as np
import gym
from gym import spaces
from envs.feature.rbf import build_features_gw_state

class GridWorld(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, shape=[9., 9.],fail_prob=0.1, goal=None, start=None, gamma=0.99,  horizon=500,
                 rew_weights=None, randomized_initial=False, direction='center', n_bases=[9, 9], CSI=None, ml=None, border_width=2):

        assert shape[0] >= 3 and shape[1] >= 3, "The grid must be at least 3x3"
        assert horizon >= 1, "The horizon must be at least 1"
        assert 0 <= fail_prob <= 1, "The probability of failure must be in [0,1]"
        self.horizon = horizon
        self.noise = fail_prob
        self.state_dim = 2
        self.action_dim = 2
        self.time_step = 1.
        self.speed = 1.
        self.direction = direction
        self.fail_prob = fail_prob
        self.ml = ml
        self.border_width = border_width
        self.t = 0
        if goal is None:
            self.goal = np.array([shape[1], shape[0]], dtype=np.float64)
        else:
            self.goal = goal

        if start is None:
            start = np.array([0, shape[0]], dtype=np.float64)
        size = np.array(shape)
        size[1] = 2*size[1]

        self.CSI = CSI
        self.size = size
        self.goal_radius = 0.5
        self.done = False
        self.start = start
        self.state = start
        self.randomized_initial = randomized_initial

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_bases[0], n_bases[1]))
        #delta_x, delta_y
        self.action_space = spaces.Box(low=np.array([-3., -3.]), high=np.array((3., 3.)))

        self.PrettyTable = None
        self.rendering = None
        self.gamma = gamma
        if self.direction == 'center':
            rew_weights = [10, 1, 1, 0]
        if rew_weights is None:
            rew_weights = [1., 10, 5, 1]

        self.rew_weights = np.array(rew_weights)
        # if np.sum(rew_weights) != 0:
            # self.rew_weights = self.rew_weights / np.sum(rew_weights)

        # gym attributes
        self.viewer = None
        self.feat_func = build_features_gw_state(self.size, n_bases, 2)
        # initialize state
        self.reset(rbf=True)


    def get_rew_features(self, state=None, clipped=False):
        features = np.zeros(4)
        if state is None:
            if self.done:
                return features
            state = self.state
        x, y = state

        if clipped:
            features[3] = -1

        if self.reached_goal(state):  # goal state
            features[2] = 1
        elif self.border_width < x < self.size[0] - self.border_width and \
                self.border_width < y < self.size[1] - self.border_width:  # slow_region
            features[1] = -1.
        elif self.direction == 'up' and y < self.size[1] / 2:
            features[1] = -1
        elif self.direction == 'down' and y > self.size[1] / 2:
            features[1] = -1
        else:
            features[0] = -1.   # fast region

        return features


    def reached_goal(self, state=None):
        if state is None:
            state = self.state

        return np.linalg.norm(state - self.goal) <= self.goal_radius

    def discretize_action(self, action):

        x, y = np.array(action).clip([-1, -1], [1, 1])
        if x < 0.5 and x > -0.5 and y >= 0:
            new_action = 0  # su
        elif x < 0.5 and x > -0.5 and y <= 0:
            new_action = 1  # giu
        elif y < 0.5 and y > -0.5 and x >= 0:
            new_action = 2  # DESTRA
        elif y < 0.5 and y > -0.5 and x <= 0:
            new_action = 3  # SINISTRA
        elif x > 0 and y > 0:
            new_action = 4  # DIAGONALE DESTRA SU
        elif x <= 0 and y > 0:
            new_action = 5  # DIAGONALE SINISTRA SU
        elif x > 0 and y <= 0:
            new_action = 6  # DIAGONALE DESTRA GIU
        elif x <= 0 and y <= 0:
            new_action= 7  # DIAGONALE SINISTRA GIU
        return np.array([new_action])


    def get_CSI_reward(self, action, rbf=False, ohe=False):
        action = self.discretize_action(action)
        return self.CSI.predict([np.concatenate((self.state, action))])



    def step(self, a, rbf=False, ohe=False):
        if self.reached_goal():
            self.t += 1
            return self.get_state(rbf=rbf, ohe=ohe), 0, 1, {'features': np.zeros(4)}

        # print(a)
        self.state += self.speed * self.time_step * np.array(a).clip([-3, -3], [3, 3])
        # Add noise
        if self.noise > 0.:
            self.state += np.random.normal(scale=self.noise, size=(2,))
        # print('s',self.state)
        if self.CSI != None and False:
            if a[0] < 0:
                a[0] = -1
            if a[0] <= -0.5:
                a[0] = -1
            elif a[0] >= 0.5:
                a[0] = 1
            else:
                a[0] = 0
            if a[1] <= -0.5:
                a[1] = -1
            elif a[1] >= 0.5:
                a[1] = 1
            else:
                a[1] = 0

        # Clip to make sure the agent is inside the grid
        state_before = self.state
        state_clip = self.state.clip([0., 0.], self.size - 1e-8)
        #punish going out?
        if self.state[0] != state_clip[0] or self.state[1] != state_clip[1]:
            clipped = True
        else:
            clipped = False

        self.state = state_clip
        features = self.get_rew_features(clipped=clipped)
        # Compute reward
        reward = np.sum(self.rew_weights * features)
        if self.CSI != None:
            reward = self.get_CSI_reward(a, rbf)

        # print(self.reached_goal())
        self.t += 1
        self.done = 1 if self.reached_goal() else 0 #or self.t >= self.horizon 
        # print(features)
        return self.get_state(rbf=rbf, ohe=ohe), reward, self.done, {'features': features, 'position': state_before}


    def get_state(self, rbf=False,ohe=False):
        if rbf or ohe:
            s = self.feat_func(self.state)
            return s

        return self.state

    def reset(self, state=None, rbf=False, ohe=False):
        self.done = False
        self.t = 0
        if state is None:
            if self.randomized_initial:
                self.state = np.copy(self.goal)
                while self.reached_goal():
                    self.state = np.random.uniform(low=[0, 0], high=self.size)
            else:
                self.state = np.copy(self.start)
        else:
            self.state = np.array(state)

        return self.get_state(rbf=rbf, ohe=ohe)

    def _render(self, mode='human', close=False, a=None):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        from gym.envs.classic_control import rendering
        ratio = self.size[0] / self.size[1]
        if self.viewer is None:
            self.viewer = rendering.Viewer(int(500 * ratio), 500)
            self.viewer.set_bounds(0, self.size[0], 0, self.size[1])

        self.viewer.draw_line((self.border_width, self.border_width), (self.size[0] - self.border_width, self.border_width))
        self.viewer.draw_line((self.border_width, self.border_width), (self.border_width, self.size[1] - self.border_width))
        self.viewer.draw_line((self.size[0] - self.border_width, self.border_width),
                              (self.size[0] - self.border_width, self.size[1] - self.border_width))
        self.viewer.draw_line((self.border_width, self.size[1] - self.border_width),
                              (self.size[0] - self.border_width, self.size[1] - self.border_width))

        if self.state is None:
            return None

        goal = self.viewer.draw_circle(radius=self.goal_radius)
        goal.set_color(0, 0.8, 0)
        goal.add_attr(rendering.Transform(translation=(self.goal[0], self.goal[1])))

        agent = self.viewer.draw_circle(radius=0.1)

        agent.set_color(.8, 0, 0)
        transform = rendering.Transform(translation=(self.state[0], self.state[1]))
        agent.add_attr(transform)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


class GridWorldAction(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, shape=[9., 9.],fail_prob=0., goal=None, start=None, gamma=0.99,  horizon=500,
                 rew_weights=None, randomized_initial=False, direction='center', n_bases=[5, 5],
                 CSI=None, ml=None, border_width=2):

        assert shape[0] >= 3 and shape[1] >= 3, "The grid must be at least 3x3"
        assert horizon >= 1, "The horizon must be at least 1"
        assert 0 <= fail_prob <= 1, "The probability of failure must be in [0,1]"
        self.horizon = horizon
        self.noise = fail_prob
        self.state_dim = 2
        self.action_dim = 2
        self.time_step = 1.
        self.speed = 1.
        self.direction = direction
        self.fail_prob = fail_prob
        self.ml = ml
        self.border_width = border_width
        self.t = 0
        if goal is None:
            self.goal = np.array([shape[1], shape[0]], dtype=np.float64)
        else:
            self.goal = goal

        if start is None:
            start = np.array([0.5, shape[0]], dtype=np.float64)
        size = np.array(shape)
        size[1] = 2*size[1]

        self.CSI = CSI
        self.size = size
        self.goal_radius = 0.5
        self.done = False
        self.start = start
        self.state = start
        self.randomized_initial = randomized_initial

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_bases[0], n_bases[1]))
        #delta_x, delta_y
        self.action_low, self.action_high = np.array([-1., -1.]), np.array([1., 1.])
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high)

        self.PrettyTable = None
        self.rendering = None
        self.gamma = gamma


        self.rew_weights = np.array(rew_weights)
        # if np.sum(rew_weights) != 0:
            # self.rew_weights = self.rew_weights / np.sum(rew_weights)

        # gym attributes
        self.viewer = None
        self.feat_func = build_features_gw_state(self.size, n_bases, 2)
        # initialize state
        self.reset(rbf=True)


    def get_rew_features(self, state=None, action=None, clipped=False):
        features = np.zeros(3)
        x, y = state
        if clipped:
            features[0] = features[1] = -1.
        elif self.border_width < x < self.size[0] - self.border_width and \
                 self.border_width < y < self.size[1] - self.border_width:  # slow_region
            features[1] = -1.
        #elif y < self.size[1] / 2 - self.border_width/2 or y > self.size[1] / 2 + self.border_width/2:  #fast region
        else:
            features[0] = -1.

        action_norm = np.linalg.norm(action)
        features[2] = -action_norm ** 2

        return features

    def reached_goal(self, state=None):
        if state is None:
            state = self.state

        x, y = state

        return x > self.size[0] - self.border_width and self.size[1] / 2 - self.border_width / 2 < y < self.size[1] / 2 + self.border_width / 2  #np.linalg.norm(state - self.goal) <= self.goal_radius

    def discretize_action(self, action):

        x, y = np.array(action).clip([-1, -1], [1, 1])
        if 0.5 > x > -0.5 and y >= 0:
            new_action = 0  # su
        elif 0.5 > x > -0.5 and y <= 0:
            new_action = 1  # giu
        elif 0.5 > y > -0.5 and x >= 0:
            new_action = 2  # DESTRA
        elif 0.5 > y > -0.5 and x <= 0:
            new_action = 3  # SINISTRA
        elif x > 0 and y > 0:
            new_action = 4  # DIAGONALE DESTRA SU
        elif x <= 0 and y > 0:
            new_action = 5  # DIAGONALE SINISTRA SU
        elif x > 0 and y <= 0:
            new_action = 6  # DIAGONALE DESTRA GIU
        elif x <= 0 and y <= 0:
            new_action = 7  # DIAGONALE SINISTRA GIU
        return np.array([new_action])

    def get_CSI_reward(self, action, rbf=False, ohe=False):
        action = self.discretize_action(action)
        return self.CSI.predict([np.concatenate((self.state, action))])

    def step(self, a, rbf=False, ohe=False):

        if self.reached_goal():
            self.t += 1
            return self.get_state(rbf=rbf, ohe=ohe), 0, 1, {'features': np.zeros(3)}


        action = np.array(a).clip(self.action_low, self.action_high)
        #print(a, action, self.state)
        # print(a)
        self.state += self.speed * self.time_step * action
        # Add noise
        if self.noise > 0.:
            self.state += np.random.normal(scale=self.noise, size=(2,))
        # print('s',self.state)

        if self.CSI != None and False:
            if a[0] < 0:
                a[0] = -1
            if a[0] <= -0.5:
                a[0] = -1
            elif a[0] >= 0.5:
                a[0] = 1
            else:
                a[0] = 0
            if a[1] <= -0.5:
                a[1] = -1
            elif a[1] >= 0.5:
                a[1] = 1
            else:
                a[1] = 0

        # Clip to make sure the agent is inside the grid
        state_before = self.state
        state_clip = self.state.clip([0., 0.], self.size)

        if not np.all(np.isclose(self.state, state_clip)):
            clipped = True
        else:
            clipped = False

        #print(clipped)

        self.state = state_clip
        features = self.get_rew_features(state=self.state, action=action, clipped=clipped)
        # Compute reward
        reward = np.sum(self.rew_weights * features)
        if self.CSI != None:
            reward = self.get_CSI_reward(a, rbf)

        # print(self.reached_goal())
        self.t += 1
        self.done = self.reached_goal() or self.t >= self.horizon
        # print(features)
        return self.get_state(rbf=rbf, ohe=ohe), reward, self.done, {'features': features, 'position': state_before}


    def get_state(self, rbf=False,ohe=False):
        if rbf or ohe:
            s = self.feat_func(self.state)
            return s

        return self.state

    def reset(self, state=None, rbf=False, ohe=False):
        self.done = False
        self.t = 0
        if state is None:
            if self.randomized_initial:
                self.state = np.copy(self.goal)
                while self.reached_goal():
                    self.state = np.random.uniform(low=[0.5, 0.5], high=self.size-0.5)
            else:
                self.state = np.copy(self.start)
        else:
            self.state = np.array(state)

        return self.get_state(rbf=rbf, ohe=ohe)

    def _render(self, mode='human', close=False, a=None):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        from gym.envs.classic_control import rendering
        ratio = self.size[0] / self.size[1]
        if self.viewer is None:
            self.viewer = rendering.Viewer(int(500 * ratio), 500)
            self.viewer.set_bounds(0, self.size[0], 0, self.size[1])

        self.viewer.draw_line((self.border_width, self.border_width), (self.size[0] - self.border_width, self.border_width))
        self.viewer.draw_line((self.border_width, self.border_width), (self.border_width, self.size[1] - self.border_width))
        self.viewer.draw_line((self.size[0] - self.border_width, self.border_width),
                              (self.size[0] - self.border_width, self.size[1] - self.border_width))
        self.viewer.draw_line((self.border_width, self.size[1] - self.border_width),
                              (self.size[0] - self.border_width, self.size[1] - self.border_width))

        if self.state is None:
            return None

        goal = self.viewer.draw_circle(radius=self.goal_radius)
        goal.set_color(0, 0.8, 0)
        goal.add_attr(rendering.Transform(translation=(self.goal[0], self.goal[1])))

        agent = self.viewer.draw_circle(radius=0.1)

        agent.set_color(.8, 0, 0)
        transform = rendering.Transform(translation=(self.state[0], self.state[1]))
        agent.add_attr(transform)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

import numpy as np
from gym import spaces
from envs.feature.rbf import build_features_gw_state
from envs.continuous_gridword import GridWorld

class GridWorld2(GridWorld):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, shape=[9., 9.], fail_prob=0., goal=None, start=None, gamma=0.99,
                 rew_weights=None, randomized_initial=False, direction='center', n_bases=[5, 5], CSI=None, ml=None):

        assert shape[0] >= 3 and shape[1] >= 3, "The grid must be at least 3x3"
        assert 0 <= fail_prob <= 1, "The probability of failure must be in [0,1]"
        self.noise = fail_prob
        self.fail_prob = fail_prob
        self.ml = ml
        if goal == None:
            self.goal = np.array([9, 4.5], dtype=np.float64)
        else:
            self.goal = goal
        if start == None:
            self.start = np.array([0, 4.5],dtype=np.float64)
        else:
            self.start = start
        size = np.array(shape)
        size[1] = size[1]

        self.CSI = CSI
        self.size = size
        self.goal_radius = 1.
        self.direction = direction
        self.done = False
        self.state = start
        self.randomized_initial = randomized_initial

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_bases[0], n_bases[1]))
        #delta_x, delta_y
        self.action_space = spaces.Box(low=np.array([-1., -1.]), high=np.array((1., 1.)))

        self.rendering = None
        self.gamma = gamma
        if rew_weights is None:
            rew_weights = [1,10,1]
            if self.direction == 'center':
                rew_weights = [1,1,1]
        self.rew_weights = np.array(rew_weights)

        # gym attributes
        self.viewer = None
        self.feat_func = build_features_gw_state(self.size, n_bases, 2)
        # initialize state
        self.reset(rbf=True)


    def get_rew_features(self, state=None):
        if state is None:
            if self.done:
                return np.zeros(3)
            state = self.state
        x, y = self.state
        features = np.zeros(3)
        if self.reached_goal(state):  # goal state
            features[2] = 1
        elif x > 2 and x < self.size[0]-2 and y > 2 and y < self.size[1]-2:  # slow_region
            features[1] = -1#np.linalg.norm(state - self.goal)
        elif self.direction == 'up' and y<self.size[1]/2:
            features[1] = -1#np.linalg.norm(state - self.goal)
        elif self.direction == 'down' and y>self.size[1]/2:
            features[1] = -1#np.linalg.norm(state - self.goal)
        else:
            features[0] = -1#np.linalg.norm(state - self.goal)   # fast region
        if not self.ml is None:
            f = np.eye(100)
            s = self.state[0] + self.state[1]*9
            return np.dot(self.ml,f[int(s)])
        # print(self.state, features)
        return features


    def reached_goal(self, state=None):
        if state is None:
            state = self.state
        return np.linalg.norm(state - self.goal) < self.goal_radius

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
            return self.get_state(rbf=rbf,ohe=ohe), 0., 1, {'features': np.zeros(3)}
        if np.random.random() < self.fail_prob:
            a = np.random.uniform(-1, 1, 2)

        self.state += np.array(a).clip([-1, -1], [1, 1])
        if self.CSI is not None and False:
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

        # Add noise
        if self.noise > 0:
            self.state += np.random.normal(scale=self.noise, size=(2,))

        # Clip to make sure the agent is inside the grid


        self.state = self.state.clip([0., 0.], self.size - 1e-8)

        # Compute reward
        features = self.get_rew_features()
        reward = np.sum(self.rew_weights * features)
        if self.CSI != None:
            reward = self.get_CSI_reward(a, rbf)
        self.done = 1 if self.reached_goal() else 0
        # print(self.done)
        return self.get_state(rbf=rbf,ohe=ohe), reward, self.done,  {'features': features}, self.state


    def get_state(self, rbf=False,ohe=False):
        if rbf or ohe:
            s = self.feat_func(self.state)
            return s
        return self.state


    def reset(self, state=None, rbf=False, ohe=False):
        self.done = False
        if state is None:
            if self.randomized_initial:
                self.state = np.copy(self.goal)
                while self.reached_goal():
                    self.state = np.random.uniform(low=[0,0], high=[self.size[0],self.size[1]])
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

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(0, self.size[0], 0, self.size[1])

        self.viewer.draw_line((1, 1), (self.size[0]-1, 1))
        self.viewer.draw_line((1, 1), ( 1, self.size[1]-1))
        self.viewer.draw_line((self.size[0] - 1, 1), (self.size[0] - 1, self.size[1] -1))
        self.viewer.draw_line(( 1, self.size[1]-1), (self.size[0] - 1, self.size[1] -1))

        if self.state is None:
            return None

        goal = self.viewer.draw_circle(radius=self.goal_radius)
        goal.set_color(0, 0.8, 0)
        goal.add_attr(rendering.Transform(translation=(self.goal[0], self.goal[1])))
        # goal.add_attr(rendering.Transform(translation=(2, 0)))
        agent = self.viewer.draw_circle(radius=0.1)

        agent.set_color(.8, 0, 0)
        transform = rendering.Transform(translation=(self.state[0], self.state[1]))
        agent.add_attr(transform)


        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
import math

import numpy as np

import gym
from gym import spaces
from envs.feature.rbf import build_features_mch_state
from gym.utils import seeding


class ContinuousMountainCar(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, horizon=1000, fail_prob=0.1, goal_position=None, start_state=None,
                 rew_weights=None, randomized_start=False, n_bases=[10, 10], rew_basis=[3, 3]):

        assert 0 <= fail_prob <= 1, "The probability of failure must be in [0,1]"
        self.horizon = horizon
        self.noise = fail_prob
        if goal_position is None:
            goal_position = 0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.randomized_start = randomized_start

        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = goal_position
        self.power = 0.0015

        if start_state is None:
            start_state = np.random.uniform(low=[-0.6, 0], high=[-0.4, 0])
        self.start_state = start_state

        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_bases[0] * n_bases[1],))

        self.size = {'position': [self.min_position, self.max_position],
                     'speed': [-self.max_speed, self.max_speed]}
        self.feat_func = build_features_mch_state(self.size, n_bases, 2)
        self.n_bases = n_bases

        self.rew_feat_func = build_features_mch_state(self.size, rew_basis, 2)
        self.rew_basis = rew_basis

        self.enabled_weights = False
        if rew_weights is not None:
            self.enabled_weights = True
            self.rew_weights = rew_weights

        self.rew_weights = rew_weights
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, rbf=False, ohe=False):

        if self.done:
            return self.get_state(rbf=rbf, ohe=ohe), 0, self.done, {'features': np.zeros(self.rew_basis[0]*self.rew_basis[1])}
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        velocity += force * self.power - 0.0025 * math.cos(3*position)
        if velocity > self.max_speed: velocity = self.max_speed
        if velocity < -self.max_speed: velocity = -self.max_speed
        position += velocity
        if position > self.max_position: position = self.max_position
        if position < self.min_position: position = self.min_position
        if position == self.min_position and velocity < 0: velocity = 0

        self.t += 1
        self.done = bool(position >= self.goal_position) or self.t >= self.horizon

        features = -self.get_rew_features()
        if not self.enabled_weights:
            reward = 0
            if bool(position >= self.goal_position):
                reward = 100.0
            reward -= (action[0] ** 2) * 0.1
        else:
            reward = (self.rew_weights * features).sum(axis=-1)

        features = self.get_rew_features()
        if self.rew_weights is None:
            reward = 0
            if bool(position >= self.goal_position):
                reward = 100.0
            reward -= (action[0] ** 2) * 0.1
        else:
            reward = (features * self.rew_weights).sum()

        self.state = np.array([position, velocity])



        return self.get_state(rbf=rbf, ohe=ohe), reward, self.done, {'features': features}

    def get_rew_features(self, state=None):
        if state is None:
            state = self.state
        if self.done:
            return np.zeros(self.rew_basis[0]*self.rew_basis[1])
        state = self.rew_feat_func(state)
        return state

    def reached_goal(self, state=None):
        if state is None:
            state = self.state
        return state[0] >= self.goal_position

    def get_state(self, rbf=False, ohe=False):
        if rbf or ohe:
            return self.feat_func(self.state)
        return self.state

    def reset(self, state=None, rbf=False, ohe=False):
        self.done = False
        self.t = 0
        if state is None:
            if self.randomized_start:
                self.state = np.array([self.goal_position, 0])
                while self.reached_goal():
                    self.state = np.random.uniform(low=[self.min_position, 0], high=[self.max_position, 0])
            else:
                self.state = np.copy(self.start_state)
        else:
            self.state = np.array(state)
        return self.get_state(rbf=rbf, ohe=ohe)

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def _render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth=40
        carheight=20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
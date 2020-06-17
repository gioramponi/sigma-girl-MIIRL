import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy import linalg as la, stats as scistat

class Twitter(gym.Env):

    def __init__(self,  horizon=np.inf, gamma=0.99, rew_weights=[0.4, 0.3, 0.3], T=100,tweet_dist=None,
                 low=0.1, high=0.6, circular_time=False, pos_dt=False):
        self.horizon = horizon
        self.gamma = gamma
        self.circular_time = circular_time
        self.pos_dt = pos_dt
        if circular_time:
            self.state_dim = 5
        else:
            self.state_dim = 4
        self.low = low
        self.high = high
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_dim, 1))
        if rew_weights is None:
            rew_weights = np.array([0.4, 0.3, 0.3])
        self.rew_weights = np.copy(rew_weights)
        if tweet_dist is None:
            tweet_dist = [(10., 0.5), (14., 0.5), (21., 1.)]
        self.tweet_dist = tweet_dist
        # initialize state
        self.seed()
        self.T = T
        self.reset()

    def get_rew_weights(self):
        return self.rew_weights

    def set_rew_weights(self, rew_weights):
        self.rew_weights = np.copy(rew_weights)

    def get_reward(self, action, state=None):
        if state is None:
            state = self.get_state()
        features = self.get_features(state=state, action=action)
        return np.dot(features, self.rew_weights)

    def sample_tweet(self, h=None):
        if h is None:
            h = self.h
        prob = np.random.uniform(self.low, self.high)
        for dist in self.tweet_dist:
            prob += scistat.norm(dist[0], dist[1]).pdf(h)
        self.retweet_prob = np.clip(prob, 0, 1)
        return self.retweet_prob

    def get_features(self, action, state=None):
        assert action in [0, 1]
        if state is None:
            state = self.get_state()

        state[0] *= action
        if not self.pos_dt:
            state[1] *= -1
        state[2] *= -1
        return state[: len(self.rew_weights)]

    def get_state(self, rbf=False, ohe=False):

        st = [self.retweet_prob, self.dt / self.T, self.nt / self.T]
        if self.circular_time:
            st += [np.sin(self.h), np.cos(self.h)]
        else:
            st += [self.h / 24]
        return np.array(st)

    def seed(self, seed=None):
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, state=None, ohe=False, rbf=False):
        self.t = 0
        self.done = False
        if state is not None:
            self.retweet_prob, self.dt, self.nt, self.h = state[:]
            assert self.nt <= self.T - self.dt
            self.tweet_buffer = [0] * self.T
            if self.dt < self.T:
                self.tweet_buffer[-self.dt-1] = 1
            if self.n_t > 1:
                indices = np.random.choice(list(range(self.T - self.dt - 1)),
                                           replace=False, size=self.nt - 1)
                for ind in indices:
                    self.tweet_buffer[ind] = 1
            return self.get_state()

        self.tweet_buffer = [0] * self.T
        self.dt = self.T
        self.nt = 0
        self.h = np.random.uniform(0, 24)
        self.retweet_prob = self.sample_tweet()
        return self.get_state()

    def step(self, action, ohe=False, rbf=False):
        if self.done:
            return np.zeros(self.state_dim), np.asscalar(0), self.done, {'features': np.zeros(len(self.rew_weights))}
        reward = self.get_reward(action=action)
        features = self.get_features(action=action)
        self.tweet_buffer.append(action)
        self.nt += action
        self.nt -= self.tweet_buffer.pop(0)
        self.t += 1
        self.h += np.random.uniform(0, 1)
        self.h %= 24
        if action == 1:
            self.dt = 0
        elif self.dt < self.T:
            self.dt += 1
        assert self.nt <= self.T - self.dt
        self.retweet_prob = self.sample_tweet(self.h)
        if self.t == self.horizon:
            self.done = True
        return self.get_state(), np.asscalar(reward), self.done, {'features': features}


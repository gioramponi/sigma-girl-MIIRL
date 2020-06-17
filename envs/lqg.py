import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy import linalg as la


class LQG(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, n=2, horizon=100, gamma=0.99, max_state=10., max_action=8., sigma_noise=0.1,
                 action_scales=[0.2, 0.05], a=None,b=None, rotation=False, theta=np.pi/6, shift=0.5):
        self.horizon = horizon
        self.gamma = gamma
        self.n = n
        self.max_pos = max_state
        self.max_action = max_action

        if sigma_noise is not None and isinstance(sigma_noise, (int,  float, complex)):
            sigma_noise = np.diag(np.ones(self.n)*sigma_noise)
        self.sigma_noise = sigma_noise

        self.A = np.eye(n)
        if rotation and n ==2:
            cos = np.cos(theta)
            sin = np.sin(theta/3)
            self.A =np.array([[cos,-sin],[sin,cos]])
        if a is not None:
            self.A = a
        if action_scales is not None and isinstance(action_scales, (int,  float, complex)):
            action_scales = np.diag(np.ones(self.n)*action_scales)
        self.B = np.diag(np.array(action_scales))
        if b is not None:
            self.B = b

        assert -1 <= shift < 1
        self.Q = np.diag(np.ones(n)) / (2*n) * (1 + shift)
        self.R = np.diag(np.ones(n)) / (2*n) * (1 - shift)

        # gym attributes
        self.viewer = None
        high = self.max_pos
        self.action_space = spaces.Box(low=-self.max_action,
                                       high=self.max_action,
                                       shape=(n, 1))
        self.observation_space = spaces.Box(low=-high, high=high, shape=(n, 1))
        # initialize state
        self.seed()
        self.reset()

    def get_rew_weights(self):
        return np.concatenate([np.diag(self.Q), np.diag(self.R)])

    def get_cost(self, x, u):
        return np.dot(x.T, np.dot(self.Q, x)) + np.dot(u.T, np.dot(self.R, u))

    def set_costs(self, Q, R):
        self.Q = np.array(Q)
        self.R = np.array(R)

    def set_dynamics(self, A, B):
        self.A = np.array(A)
        self.B = np.array(B)

    def get_features(self, x, u):
        return -np.concatenate([x.reshape((self.n,))**2, u.reshape((self.n,))**2])

    def _add_noise(self):
        return np.random.multivariate_normal(np.zeros(self.n), self.sigma_noise, 1).T

    def step(self, action, render=False):
        if self.done:
            return self.get_state(), -np.asscalar(0), self.done, {'features': np.zeros(2*self.n)}
        u = np.clip(action, -self.max_action, self.max_action)
        u = u.reshape((self.n, 1))
        noise = self._add_noise()
        xn = np.dot(self.A, self.state) + np.dot(self.B, u) + noise.reshape(-1, 1)
        xn = np.clip(xn, -self.max_pos, self.max_pos)
        cost = self.get_cost(self.state, u)
        self.t += 1
        if self.t == self.horizon:
            self.done = True
        self.state = np.array(xn)
        features = self.get_features(self.state, u)
        return self.get_state(), -np.asscalar(cost), self.done, {'features': features}

    def seed(self, seed=None):
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, state=None, random_init=False, different_scales=False):
        if random_init:
            self.state = np.random.uniform(-self.max_pos, self.max_pos, self.n)
        else:
            self.state = np.ones(self.n) * np.random.choice([-self.max_pos, self.max_pos], self.n)
        if self.n > 1 and different_scales:
            self.state[self.n-1] = self.state[self.n-1] / 2
        #self.state = np.array([self.max_pos, 0])
        self.state = self.state.reshape((-1, 1))
        self.t = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        return self.state

    def _render(self, mode='human', close=False):
        if self.n > 2:
            return
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 600

        world_width = (self.max_pos * 2) * 2
        world_height = (self.max_pos * 2) * 2
        scale = screen_width / world_width
        scale_y = screen_height / world_height
        bally = screen_height/2
        ballradius = 3

        if self.viewer is None:
            clearance = 0  # y-offset
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            mass = rendering.make_circle(ballradius * 2)
            mass.set_color(.8, .3, .3)
            mass.add_attr(rendering.Transform(translation=(0, clearance)))
            self.masstrans = rendering.Transform()
            mass.add_attr(self.masstrans)
            self.viewer.add_geom(mass)
            self.track = rendering.Line((0, screen_height/2), (screen_width, screen_height/2))
            self.track.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(self.track)
            zero_line = rendering.Line((screen_width / 2, 0),
                                       (screen_width / 2, screen_height))
            zero_line.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(zero_line)

        x = self.state[0]
        if self.n==2:
            y = self.state[1]
            bally = y * scale_y + screen_height / 2.0
        ballx = x * scale + screen_width / 2.0
        self.masstrans.set_translation(ballx, bally)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def computeP(self, K, tolerance=0.001):
        """
               This function computes the Riccati equation associated to the LQG
               problem.
               Args:
                   K (matrix): the matrix associated to the linear controller K * x
               Returns:
                   P (matrix): the Riccati Matrix
        """
        I = np.eye(self.Q.shape[0], self.Q.shape[1])
        if np.array_equal(self.A, I) and np.array_equal(self.B, I) and False:
            P = (self.Q + np.dot(K.T, np.dot(self.R, K))) / (I - self.gamma *
                                                             (I + 2 * K + K **
                                                              2))
        else:
            converged = False
            P = np.eye(self.Q.shape[0], self.Q.shape[1])
            t = 0
            while not converged:
                f1 = self.Q + self.gamma * np.dot(self.A.T, np.dot(P, self.A))
                f2 = self.gamma * np.dot(K.T, np.dot(self.B.T, np.dot(P, self.A)))
                f3 = self.gamma * np.dot(self.A.T, np.dot(P, np.dot(self.B, K)))
                f4 = self.gamma * np.dot(K.T, np.dot(self.B.T, np.dot(P, np.dot(self.B, K)))) + np.dot(K.T,
                                                                                                       np.dot(self.R,
                                                                                                              K))
                Pnew = f1 + f2 + f3 + f4
                t += 1
                if np.isnan(Pnew).any():
                    print("Nan Found")
                converged = np.max(np.abs(P - Pnew)) < tolerance
                P = Pnew
        return P

    def computeOptimalK(self):
        """
        This function computes the optimal linear controller associated to the
        LQG problem (u = K * x).

        Returns:
            K (matrix): the optimal controller bv

        """
        P = la.solve_discrete_are(self.A, self.B, self.Q, self.R)
        K = -np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(self.B.T, P), self.B) + self.R), self.B.T), P), self.A)
        return K

    def computeJ(self, K, Sigma, n_random_x0=100, tolerance=0.001, random_init=False):

        if isinstance(K, (int,  float, complex)):
            K = np.array([K]).reshape(1, 1)
        if isinstance(Sigma, (int, float, complex)):
            Sigma = np.array([Sigma]).reshape(1, 1)

        P = self.computeP(K, tolerance)
        J = 0.0
        for i in range(n_random_x0):
            s = self.reset(random_init=random_init)
            x0 = self.get_state()
            J -= np.dot(x0.T, np.dot(P, x0)) \
                + (1 / (1 - self.gamma)) * \
                np.trace(np.dot(
                    Sigma, (self.R + self.gamma * np.dot(self.B.T,
                                                         np.dot(P, self.B)))))
        J /= n_random_x0
        return J

    def computeQFunction(self, x, u, K, Sigma, n_random_xn=100, tolerance=0.001):
        """
        This function computes the Q-value of a pair (x,u) given the linear
        controller Kx + epsilon where epsilon \sim N(0, Sigma).
        Args:
            x (int, array): the state
            u (int, array): the action
            K (matrix): the controller matrix
            Sigma (matrix): covariance matrix of the zero-mean noise added to
            the controller action
            n_random_xn: the number of samples to draw in order to average over
            the next state

        Returns:
            Qfun (float): The Q-value in the given pair (x,u) under the given
            controller

        """
        if isinstance(x, (int,  float, complex)):
            x = np.array([x])
        if isinstance(u, (int,  float, complex)):
            u = np.array([u])
        if isinstance(K, (int,  float, complex)):
            K = np.array([K]).reshape(1, 1)
        if isinstance(Sigma, (int, float, complex)):
            Sigma = np.array([Sigma]).reshape(1, 1)

        P = self.computeP(K, tolerance)
        Qfun = 0
        for i in range(n_random_xn):
            noise = self._add_noise()
            action_noise = np.random.multivariate_normal(
                np.zeros(Sigma.shape[0]), Sigma, 1).T
            nextstate = np.dot(self.A, x) + np.dot(self.B,
                                                   u + action_noise) + noise
            Qfun -= np.dot(x.T, np.dot(self.Q, x)) + \
                np.dot(u.T, np.dot(self.R, u)) + \
                self.gamma * np.dot(nextstate.T, np.dot(P, nextstate)) + \
                (self.gamma / (1 - self.gamma)) * \
                np.trace(np.dot(Sigma,
                                self.R + self.gamma *
                                np.dot(self.B.T, np.dot(P, self.B))))
        Qfun = np.asscalar(Qfun) / n_random_xn
        return Qfun

    def computeVFunction(self, x, K, Sigma, n_random_xn=100, tolerance=0.001):
        """
        This function computes the Q-value of a pair (x,u) given the linear
        controller Kx + epsilon where epsilon \sim N(0, Sigma).
        Args:
            x (int, array): the state
            u (int, array): the action
            K (matrix): the controller matrix
            Sigma (matrix): covariance matrix of the zero-mean noise added to
            the controller action
            n_random_xn: the number of samples to draw in order to average over
            the next state

        Returns:
            Qfun (float): The Q-value in the given pair (x,u) under the given
            controller

        """
        if isinstance(x, (int, float, complex)):
            x = np.array([x])
        if isinstance(K, (int, float, complex)):
            K = np.array([K]).reshape(1, 1)
        if isinstance(Sigma, (int, float, complex)):
            Sigma = np.array([Sigma]).reshape(1, 1)

        P = self.computeP(K, tolerance)
        Vfun = 0
        for i in range(n_random_xn):
            action_noise = np.random.multivariate_normal(
                np.zeros(Sigma.shape[0]), Sigma, 1).T
            u = action_noise + np.dot(K, x)
            noise = self._add_noise()
            nextstate = np.dot(self.A, x) + np.dot(self.B,
                                                   u) + noise

            Vfun -= np.dot(x.T, np.dot(self.Q, x)) + \
                    np.dot(u.T, np.dot(self.R, u)) + \
                    self.gamma * np.dot(nextstate.T, np.dot(P, nextstate)) + \
                    (self.gamma / (1 - self.gamma)) * \
                    np.trace(np.dot(Sigma,
                                    self.R + self.gamma *
                                    np.dot(self.B.T, np.dot(P, self.B))))
        Qfun = np.asscalar(Vfun) / n_random_xn
        return Qfun


class LQGSimple(LQG):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, n=2, horizon=100, gamma=0.99, max_state=10., max_action=8., sigma_noise=0.1,
                 A=None, B=None, R=None, Q=None):
        self.horizon = horizon
        self.gamma = gamma
        self.n = n
        self.max_pos = max_state
        self.max_action = max_action

        if sigma_noise is not None and isinstance(sigma_noise, (int,  float, complex)):
            sigma_noise = np.eye(self.n) * sigma_noise
        self.sigma_noise = sigma_noise

        if A is not None:
            self.A = self._convert_parameters(A)
        else:
            self.A = np.eye(self.n)

        if B is not None:
            self.B = self._convert_parameters(B)
        else:
            self.B = np.eye(self.n)

        if Q is not None:
            self.Q = self._convert_parameters(Q)
        else:
            self.Q = np.eye(self.n) / (2 * self.n)

        if R is not None:
            self.R = self._convert_parameters(R)
        else:
            self.R = np.eye(self.n) / (2 * self.n)

        # gym attributes
        self.viewer = None
        high = self.max_pos
        self.action_space = spaces.Box(low=-self.max_action,
                                       high=self.max_action,
                                       shape=(n, 1))
        self.observation_space = spaces.Box(low=-high, high=high, shape=(n, 1))
        # initialize state
        self.seed()
        self.reset()

    def _convert_parameters(self, X):
        if isinstance(X, (int, float, complex)):
            return np.eye(self.n) * X

        X = np.array(X)
        if np.ndim(X) == 1:
            return np.diag(X)

        if np.ndim(X) == 2:
            return X

    def reset(self, state=None, random_init=False):
        if random_init:
            self.state = np.random.uniform(-self.max_pos, self.max_pos, self.n)
        else:
            self.state = np.ones(self.n) * np.random.choice([-self.max_pos, self.max_pos], self.n)

        self.state = self.state.reshape((-1, 1))
        self.t = 0
        self.done = False
        return self.get_state()

    def computeJ(self, K, Sigma, n_random_x0=100, tolerance=0.001, random_init=True):

        if isinstance(K, (int,  float, complex)):
            K = np.array([K]).reshape(1, 1)
        if isinstance(Sigma, (int, float, complex)):
            Sigma = np.array([Sigma]).reshape(1, 1)

        P = self.computeP(K, tolerance)
        J = 0.0
        for i in range(n_random_x0):
            self.reset(random_init=random_init)
            x0 = self.get_state()
            J -= np.dot(x0.T, np.dot(P, x0)) \
                + (1 / (1 - self.gamma)) * \
                np.trace(np.dot(
                    Sigma, (self.R + self.gamma * np.dot(self.B.T,
                                                         np.dot(P, self.B)))))
        J /= n_random_x0
        return J





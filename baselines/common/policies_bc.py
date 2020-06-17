import tensorflow as tf
import numpy as np
from baselines.common import tf_util
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.models import get_network_builder

import gym

def boltzmann(logits, beta=1.0):

    qs = beta * logits
    qs -= tf.reduce_max(qs, axis=1)
    qs = tf.exp(qs)
    return qs / tf.reduce_sum(qs, axis=1)

def build_gradient(logits, vars, ob_ph, n_actions=3):

    action_ph = tf.placeholder(tf.int32, [None], name='targets_placeholder')
    action_selected = tf.one_hot(action_ph, n_actions)
    # out = tf.reduce_sum(tf.reduce_sum(tf.log(self.logits+1e-5)*action_selected, axis=1))
    out = tf.reduce_sum(tf.log(tf.reduce_sum(logits * action_selected, axis=1)))
    gradients = tf.gradients(out, vars)
    compute_gradients = tf_util.function(
        inputs=[ob_ph, action_ph],
        outputs=gradients
    )
    return compute_gradients

class Policy(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, observations, action_space, latent, optimizer=None, sess=None, train=True, beta=1.0,
                 l2=0., lr=0.001, init_scale=0.01, init_bias=0.0, trainable_variance=True, trainable_bias=True,
                 init_logstd=0., scope_name="pi", clip=None, class_weights=None, **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        latent = tf.layers.flatten(latent)

        self.action_space = action_space
        self.pdtype = make_pdtype(action_space)
        self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=init_scale,
                                                    init_bias=init_bias,
                                                    trainable_variance=trainable_variance,
                                                    trainable_bias=trainable_bias,
                                                    init_logstd=init_logstd,
                                                    clip=clip)  # init_bias=0.0

        self.stochastic = tf.placeholder(dtype=tf.bool, shape=())
        self.action = tf_util.switch(self.stochastic, self.pd.sample(), self.pd.mode())
        self.neglogp = self.pd.neglogp(self.action)
        if beta == 1.0:
            self.logits = tf.nn.softmax(self.pd.flatparam())
        else:
            self.logits = boltzmann(self.pd.flatparam(), beta=beta)
        if optimizer is None:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        else:
            self.optimizer = optimizer
        self.sess = sess
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
        if train:
            self.gamma = l2
            self._build_train(class_weights=class_weights)

        try:
            self.action_ph = tf.placeholder(tf.int64, [None], name='targets_placeholder')
            action_selected = tf.one_hot(self.action_ph, self.action_space.n)

        #out = tf.reduce_sum(tf.reduce_sum(tf.log(self.logits+1e-5)*action_selected, axis=1))
            out = tf.reduce_sum(tf.log(tf.reduce_sum(self.logits*action_selected, axis=1)))
            gradients = tf.gradients(out, self.vars)
        except:
            self.action_ph = tf.placeholder(dtype=tf.float32, shape=(None,) +action_space.shape, name='targets_placeholder')
            gradients = tf.gradients(-self.pd.neglogp(self.action_ph), self.vars)

        self.compute_gradients = tf_util.function(
            inputs=[self.X, self.action_ph],
            outputs=gradients
        )
        '''self.compute_cont_gradients = tf_util.function(
            inputs=[self.X, self.action_ph],
            outputs=tf.gradients(-self.pd.neglogp(self.action_ph), self.vars)
        )'''
        self.debug = tf_util.function(
            inputs=[self.X, self.action_ph],
            outputs=[gradients, self.logits, self.action_ph]
        )
        self.set_from_flat = tf_util.SetFromFlat(self.vars)

        accuracy = tf.reduce_mean(tf.cast(tf.math.equal(self.action, self.action_ph), tf.float32))
        entropy = tf.reduce_mean(self.pd.entropy())
        self.evaluate = tf_util.function(
            inputs=[self.X, self.action_ph, self.stochastic],
            outputs=[accuracy, entropy]
        )
        self.pdf = tf.exp(self.pd.logp(self.action_ph))

    def _build_train(self, class_weights=None):
        self.action_ph = tf.placeholder(tf.int32, [None], name='targets_placeholder')
        action_selected = tf.one_hot(self.action_ph, self.action_space.n)
        if class_weights is not None:
            sample_weights = tf.reduce_sum(tf.multiply(action_selected, class_weights), 1)
            loss = tf.losses.sparse_softmax_cross_entropy(
                labels=self.action_ph,
                logits=self.pd.flatparam(),
                weights=sample_weights
            )
        else:
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                _sentinel=None,
                labels=self.action_ph,
                logits=self.pd.flatparam(),
                name=None,
            ))

        if self.gamma != 0.:
            regularizer = tf.nn.l2_loss(self.pd.flatparam())
            loss = tf.reduce_mean(loss + self.gamma * regularizer)
        self.optimize_expression = self.optimizer.minimize(loss, var_list=self.vars)
        self.fit = tf_util.function(
            inputs=[self.X, self.action_ph],
            outputs=[loss],
            updates=[self.optimize_expression]
        )

    # def fit(self, observations, actions):
    #     sess = self.sess or tf.get_default_session()
    #     _, summary = sess.run([self.optimize_expression, self.merged], feed_dict={
    #         self.X: adjust_shape(self.X, observations), self.action_ph: adjust_shape(self.action_ph, actions)})
    #     return summary

    def _evaluate(self, variables, observation, stochastic, **extra_feed):
        sess = self.sess or tf.get_default_session()
        feed_dict = {self.X: adjust_shape(self.X, observation),
                     self.stochastic: stochastic}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation, stochastic=False, qs=False, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        a, state, neglogp, logits, q_values = self._evaluate([self.action, self.state, self.neglogp, self.logits, self.pd.flatparam()], observation, stochastic, **extra_feed)
        if state.size == 0:
            state = None
        if qs:
            return logits, a, state, neglogp, q_values

        return logits, a, state, neglogp

    def prob(self, observation, a):
        sess = self.sess or tf.get_default_session()
        feed_dict = {self.X: adjust_shape(self.X, observation),
                     self.action_ph: adjust_shape(self.action_ph, a)}
        return sess.run([self.pdf], feed_dict)[0]
        #return self._evaluate([self.pd.logp(a)], observation)

    def save(self, save_path):
        tf_util.save_variables(save_path)

    def get_weights(self, layer_wise=False):
        sess = self.sess or tf.get_default_session()
        if not layer_wise:
            layers = sess.run(self.vars)
            weights = []
            for layer in layers:
                weights.append(layer.ravel())
            return np.concatenate(weights)
        else:
            return sess.run(self.vars)

    def set_theta(self, weights):
        self.set_weights(weights)

    def set_weights(self, weights):
        self.set_from_flat(weights)

    def load_parallel(self, load_path):
        tf_util.load_variables_parallel(load_path, variables=self.vars)

    def load(self, load_path,extra_vars=None):
        tf_util.load_variables(load_path, variables=self.vars, extra_vars=extra_vars)

    def load_state(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)


def build_policy(ob_space, ac_space, policy_network, normalize_observations=False,
                 sess=None, train=True, beta=1.0, l2=0., lr=0.001,
                 init_scale =0.01, init_bias=0.0, trainable_variance=True,
                 trainable_bias=True,
                 init_logstd=0., clip=None, class_weights=None, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(scope_name="pi", nbatch=None, nsteps=None, sess=sess, observ_placeholder=None):

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)

        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            policy_latent, recurrent_tensors = policy_network(encoded_x)

            if recurrent_tensors is not None:
                # recurrent architecture, need a few more steps
                nenv = nbatch // nsteps
                assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                extra_tensors.update(recurrent_tensors)

        policy = Policy(
            observations=X,
            action_space=ac_space,
            latent=policy_latent,
            sess=sess,
            train=train,
            beta=beta,
            l2=l2,
            lr=lr,
            init_scale=init_scale,
            init_bias=init_bias,
            trainable_variance=trainable_variance,
            trainable_bias=trainable_bias,
            init_logstd=init_logstd,
            scope_name=scope_name,
            clip=clip,
            class_weights=class_weights,
            **extra_tensors
        )
        return policy

    return policy_fn


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms


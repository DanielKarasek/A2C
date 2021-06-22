import gym.spaces as spaces

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow_probability as tfp


def create_pd(net, action_space, reuse=tf.AUTO_REUSE):
    if isinstance(action_space, spaces.Box):
        return gaussian_PD(net, [action_space.low[0], action_space.high[0]], reuse)
    if isinstance(action_space, spaces.Discrete):
        return softmax_PD(net, action_space.n, reuse)


def action_space_type(action_space):
    if action_space.dtype.type is np.float32:
        return tf.float32
    if action_space.dtype.type is np.int64:
        return tf.int32


def gaussian_PD(X, constraints, reuse=tf.AUTO_REUSE):
    with tf.variable_scope("policy", reuse=reuse):
        mu = layers.fully_connected(X, 1, activation_fn=tf.nn.tanh)
        sigma = layers.fully_connected(X, 1, activation_fn=tf.nn.softplus)
        mu = tf.squeeze(mu)
        sigma = tf.squeeze(sigma+1e-5)
        distribution = Gauss(loc=mu, scale=sigma, constraints=constraints,)
        return distribution


def softmax_PD(X, num_outputs, reuse=tf.AUTO_REUSE):

    with tf.variable_scope("policy", reuse=reuse):
        logits = layers.fully_connected(X, num_outputs, activation_fn=None)
        probabilities = tf.nn.softmax(logits)
        distribution = tfp.distributions.Categorical(probs=probabilities)
        return distribution


class Gauss(tfp.distributions.Normal):

    def __init__(self, constraints, **kwargs):
        super().__init__(**kwargs)
        self.constraints = constraints
        print(self.constraints)

    def sample(self, **kwargs):
        return tf.clip_by_value(super().sample(**kwargs), self.constraints[0], self.constraints[1])

    def entropy(self, name="entropy", **kwargs):
        return self.stddev() + 0.6

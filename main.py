from multiprocessing import cpu_count

import tensorflow as tf
import numpy as np

import tensorflow.contrib.layers as layers

from Nets import get_network_builder
from Runner import VectEnvsRunner
import utils


class Schedule:
    def __init__(self, total_steps, lr_start):
        self.total_steps = total_steps
        self.lr_start = lr_start
        self.curr_step = 0

    def get_value(self, n=1):
        curr_value = self.lr_start * (1 - (self.curr_step / self.total_steps))
        self.curr_step += n
        return curr_value


class Model:
    def __init__(self,
                 net_builder,
                 actions_space,
                 inp_shape,
                 pol_entropy_ratio,
                 nenvs,
                 n_step,
                 schedule,
                 decay,
                 sess,
                 ):
        batch_size = nenvs * n_step

        ph_obs = tf.placeholder(shape=(batch_size, *inp_shape),
                                dtype=tf.float32,
                                name="observations_placeholder",
                                )
        ph_obs_step = tf.placeholder(shape=(nenvs, *inp_shape),
                                     dtype=tf.float32,
                                     name="observations_step_placeholder")
        ph_targets = tf.placeholder(shape=batch_size,
                                    dtype=tf.float32,
                                    name="targets_placeholder"
                                    )
        ph_advantage = tf.placeholder(shape=batch_size,
                                      dtype=tf.float32,
                                      name="advantage_placeholder"
                                      )
        # todo:change dtype in case of continuous action space
        ph_actions_taken = tf.placeholder(shape=batch_size,
                                          dtype=utils.action_space_type(actions_space),
                                          name="actions_taken_placeholder",
                                          )
        ph_LR = tf.placeholder(shape=[],
                               dtype=tf.float32,
                               name="learning_rate",
                               )

        common_net = net_builder(ph_obs)
        common_net_step = net_builder(ph_obs_step)

        def create_policy_state_value(X, action_space, sample=False,):
            # todo:change create in case of continuous action space
            pol = utils.create_pd(X, action_space, reuse=tf.AUTO_REUSE)

            with tf.variable_scope("state_value", reuse=tf.AUTO_REUSE):
                state_val = layers.fully_connected(inputs=X,
                                                   num_outputs=1,
                                                   activation_fn=None
                                                   )
                state_val = tf.squeeze(state_val)
            if sample:
                return pol.sample(), state_val
            else:
                return pol, state_val

        policy, state_value = create_policy_state_value(common_net,
                                                        actions_space,
                                                        sample=False,
                                                        )

        action_step, state_value_step = create_policy_state_value(common_net_step,
                                                                  actions_space,
                                                                  sample=True,
                                                                  )

        policy_entropy = policy.entropy()
        policy_entropy = tf.reduce_mean(policy_entropy, 0)

        policy_loss = -policy.log_prob(ph_actions_taken) * ph_advantage - pol_entropy_ratio * policy_entropy
        # policy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(taken_actions_one_hot, logits) *\
        #               ph_advantage - tf.stop_gradient(pol_entropy_ratio * policy_entropy)
        value_loss = tf.square(ph_targets - state_value) * 0.5

        policy_loss = tf.reduce_mean(policy_loss, 0)
        value_loss = tf.reduce_mean(value_loss, 0)
        loss = policy_loss + value_loss

        optimizer = tf.train.RMSPropOptimizer(learning_rate=ph_LR, decay=decay)
        minimize = optimizer.minimize(loss)

        def step(obs, ):
            return sess.run([action_step, state_value_step],
                            feed_dict={ph_obs_step: obs})

        def predict_values_step(obs, ):
            return sess.run(state_value_step,
                            feed_dict={ph_obs_step: obs})

        def predict_values(obs, ):
            return sess.run(state_value,
                            feed_dict={ph_obs: obs})

        def train(obs, actions_taken, targets, advantage, ):
            return sess.run([minimize, state_value, value_loss, policy_loss, policy_entropy],
                            feed_dict={ph_obs: obs,
                                       ph_actions_taken: actions_taken,
                                       ph_targets: targets,
                                       ph_advantage: advantage,
                                       ph_LR: schedule.get_value(batch_size)},
                            )

        self.train = train
        self.step = step
        self.predict_values_step = predict_values_step
        self.predict_values = predict_values


class TestModel:
    def __init__(self, action_space, nenvs):
        self.action_space = action_space
        self.nenvs = nenvs

    def step(self, obs, ):
        return [np.random.randint(0, self.action_space, self.nenvs), np.ones(self.nenvs) * 50]

    def predict_values_step(self, obs, ):
        return np.ones(self.nenvs)


def test_main():
    gym_id = "CartPole-v1"
    n_step = 8
    gamma = 0.99
    cpu_total = cpu_count()
    total_workers = (cpu_total * 3) // 4
    runner = VectEnvsRunner(n_step, gym_id, total_workers, gamma)
    runner.model = TestModel(runner.action_space(), total_workers)
    return runner


def main():
    gym_id = "PongNoFrameskip-v4"
    # gym_id = "CartPole-v0"
    # gym_id = "MountainCarContinuous-v0"
    n_step = 5
    lr = 7e-4
    # lr = 2e-2
    decay = 0.99
    gamma = 0.99
    pol_entropy = 0.01
    cpu_total = cpu_count()
    total_workers = 16
    total_steps = 80e6
    net_build = get_network_builder("conv2fully")(scale=True, reuse=tf.AUTO_REUSE)
    # net_build = get_network_builder("fully_connected_net")(hiddens=(8,), reuse=tf.AUTO_REUSE)

    with tf.Session() as sess:
        runner = VectEnvsRunner(n_step, gym_id, total_workers, gamma)
        model = Model(net_build,
                      runner.action_space(),
                      runner.observation_space(),
                      pol_entropy,
                      total_workers,
                      n_step,
                      Schedule(total_steps, lr),
                      decay,
                      sess)
        runner.model = model
        sess.run(tf.global_variables_initializer())
        for step in range(int(total_steps // n_step // total_workers)):

            data = runner.get_data()

            log = model.train(*data)
            if not step % 100:
                print(log[1][0], " ", log[2:])
        print("done")


if __name__ == '__main__':
    # run = test_main()
    main()

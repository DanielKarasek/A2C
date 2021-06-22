import numpy as np

from envs_vectorizer import VectEnv


def get_discounted(rs, next_val, dones, gamma):
    r = next_val
    disc_rewards = np.zeros(len(rs), dtype=np.float32)
    for idx, (rew, done) in enumerate(zip(rs[::-1], dones[::-1])):
        r = rew + r*gamma*(1-done)
        disc_rewards[idx] = r
    return disc_rewards


class VectEnvsRunner:
    # todo: improve work with model
    def __init__(self, n_step, env_id, total_workers, gamma):
        self.env = VectEnv(env_id, total_workers)
        self.n_step = n_step
        self.total_envs = total_workers
        self.batch_size = self.n_step * self.total_envs
        self.observation_shape = (self.batch_size, *self.observation_space())
        self.obs = self.env.reset()
        self.dones = np.zeros(self.total_envs)
        self.gamma = gamma
        self.model = None

    def observation_space(self):
        return self.env.observation_space().shape

    def action_space(self):
        return self.env.action_space()

    def get_data(self):
        obses, actions, rewards, dones, values = [], [], [], [], []
        for _ in range(self.n_step):
            new_actions, new_vals = self.model.step(self.obs)

            self.env.render(0)

            values.append(new_vals)
            obses.append(self.obs)
            actions.append(new_actions)

            new_obs, new_rews, new_dones, _ = self.env.step(new_actions)

            rewards.append(new_rews)
            dones.append([float(new_done) for new_done in new_dones])

            self.obs = new_obs

        # todo: change actions dtype for continuous
        obses = np.asarray(obses, np.float32).swapaxes(0, 1).reshape(self.observation_shape)
        actions = np.asarray(actions, np.float32).swapaxes(0, 1).flatten()
        values = np.asarray(values, np.float32).swapaxes(0, 1).flatten()
        rewards = np.asarray(rewards, np.float32).swapaxes(0, 1)
        dones = np.asarray(dones, np.float32).swapaxes(0, 1)
        next_vals = self.model.predict_values_step(self.obs)

        for idx, (row_rs, row_dones) in enumerate(zip(rewards, dones)):
            rewards[idx] = get_discounted(row_rs, next_vals[idx], row_dones, self.gamma)

        targets = rewards.flatten()
        advantages = targets - values

        return obses, actions, targets, advantages

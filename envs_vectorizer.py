from multiprocessing import Process, Pipe, current_process
from Wrappers import make_atari, wrap_deepmind
import gym
import numpy as np


def make_env(env_id, env_type):
    if env_type == "atari":
        env = make_atari(env_id)
        env = wrap_deepmind(env, frame_stack=True)
    else:
        env = gym.make(env_id)
    return env


def get_env_prob_type(env_id):
    env_dict = {}
    for env in gym.envs.registry.all():
        env_dict[env.id] = env.entry_point[9:].split(":")[0]
    env = make_env(env_id, env_dict[env_id])
    return env, env_dict[env_id]


# todo: upravit pro continuous
def worker(env_id, remote):
    env, _ = get_env_prob_type(env_id)
    try:
        while True:
            todo, args = remote.recv()
            if todo == "step":
                obs, r, d, i = env.step(args)
                if d:
                    obs = env.reset()
                remote.send((obs, r, d, i))
            elif todo == "reset":
                remote.send(env.reset())
            elif todo == "render":
                env.render()
            elif todo == "close":
                remote.close()
            elif todo == "get_obs_shape":
                remote.send(env.observation_space)
            elif todo == "get_action":
                remote.send(env.action_space)
            else:
                print("wrong remote command")
    except KeyboardInterrupt:
        print(f"{current_process()} got keyboard interrupted")


class VectEnv:
    def __init__(self, env_id, total_workers):
        self.env_num = total_workers
        remotes = (Pipe() for _ in range(self.env_num))
        self.parent_rems, worker_rems = [], []
        for remote_pair in remotes:
            self.parent_rems.append(remote_pair[0])
            worker_rems.append(remote_pair[1])

        processes = [Process(target=worker, args=(env_id, worker_rem)) for worker_rem in worker_rems]
        for process, worker_rem in zip(processes, worker_rems):
            process.start()
            worker_rem.close()

    def __len__(self):
        return self.env_num

    def step(self, actions):
        obs, rewards, dones, infos = [], [], [], []
        for remote, action in zip(self.parent_rems, actions):
            remote.send(("step", action))
        for remote in self.parent_rems:
            n_o, n_r, n_d, n_i = remote.recv()
            obs.append(n_o)
            rewards.append(n_r)
            dones.append(n_d)
            infos.append(n_i)
        return obs, rewards, dones, infos

    def reset(self):
        data = []
        for remote in self.parent_rems:
            remote.send(("reset", None))
        for remote in self.parent_rems:
            data.append(remote.recv())
        return data

    def render(self, nth):
        self.parent_rems[nth].send(("render", None))

    def observation_space(self):
        self.parent_rems[0].send(("get_obs_shape", None))
        return self.parent_rems[0].recv()

    def action_space(self):
        self.parent_rems[0].send(("get_action", None))
        return self.parent_rems[0].recv()

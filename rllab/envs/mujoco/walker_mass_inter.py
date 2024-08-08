import random

import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.misc.overrides import overrides


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param


class WalkerMassInter(MujocoEnv, Serializable):

    FILE = 'walker2d.xml'

    def __init__(
            self,
            *args, **kwargs):
        self.goals = []
        self.goal = None
        self.init_mass = None
        for i in range(100):
            prob = random.random()  # np.random.uniform()
            if prob >= 0.5:
                g = random.uniform(0, 0.5)
            else:
                g = random.uniform(3.0, 3.5)  # 3.0 - 0.5 = 2.5
            self.goals.append(g)

        self.ctrl_cost_coeff = 1e-2
        super(WalkerMassInter, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
        self.reset()

    def sample_goals(self, num_goals):
        return np.random.choice(self.goals, num_goals)

    def sample_mass(self, mass_multiplier):
        if self.init_mass is None:
            self.init_mass = self.model.body_mass

        mass_size_ = np.prod(self.model.body_mass.shape)

        body_mass_multiplyers = np.array([mass_multiplier for _ in range(mass_size_)])
        body_mass_multiplyers = np.array(1.5) ** body_mass_multiplyers
        body_mass_multiplyers = np.array(body_mass_multiplyers).reshape(self.model.body_mass.shape)

        return self.init_mass * body_mass_multiplyers

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * \
                    np.sum(np.square(action / scaling))
        forward_reward = self.get_body_comvel("torso")[0]
        alive_bonus = 1.0
        reward = forward_reward - ctrl_cost + alive_bonus
        qpos = self.model.data.qpos
        done = not (qpos[0] > 0.8 and qpos[0] < 2.0
                    and qpos[2] > -1.0 and qpos[2] < 1.0)
        return Step(next_obs, reward, done)

    @overrides
    def log_diagnostics(self, paths, prefix=''):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))

    @overrides
    def reset(self, init_state=None, reset_args=None, **kwargs):
        if reset_args is not None:
            self.goal = reset_args
        elif self.goal is None:
            self.goal = self.sample_goals(1)[0]
        self.model.body_mass = self.sample_mass(self.goal)
        self.reset_mujoco(init_state)
        self.model.data.qpos = self.init_qpos + np.random.uniform(low=-.005, high=.005, size=self.model.nq).reshape(-1, 1)
        self.model.data.qvel = self.init_qvel + np.random.uniform(low=-.005, high=.005, size=self.model.nv).reshape(-1, 1)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()

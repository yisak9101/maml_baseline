import random

import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.misc.overrides import overrides
import math

# states: [
# 0: z-coord,
# 1: x-coord (forward distance),
# 2: forward pitch along y-axis,
# 6: z-vel (up = +),
# 7: xvel (forward = +)


class HopperMassInter(MujocoEnv, Serializable):

    FILE = 'hopper.xml'

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

        self.alive_coeff = 1
        self.ctrl_cost_coeff = 0.01
        super(HopperMassInter, self).__init__(*args, **kwargs)
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

    @overrides
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos[0:1].flat,
            self.model.data.qpos[2:].flat,
            np.clip(self.model.data.qvel, -10, 10).flat,
            np.clip(self.model.data.qfrc_constraint, -10, 10).flat,
            self.get_body_com("torso").flat,
        ])

    @overrides
    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        vel = self.get_body_comvel("torso")[0]
        reward = vel + self.alive_coeff - \
            0.5 * self.ctrl_cost_coeff * np.sum(np.square(action / scaling))
        # state = self._state
        # notdone = np.isfinite(state).all() and \
        #     (np.abs(state[3:]) < 100).all() and (state[0] > .7) and \
        #     (abs(state[2]) < .2)
        done = False
        return Step(next_obs, reward, done)

    @overrides
    def reset(self, init_state=None, reset_args=None, **kwargs):
        if reset_args is not None:
            self.goal = reset_args
        elif self.goal is None:
            self.goal = self.sample_goals(1)[0]
        print(f"Goal: {self.goal}")
        self.model.body_mass = self.sample_mass(self.goal)
        self.reset_mujoco(init_state)
        self.model.data.qpos = self.init_qpos + np.random.uniform(low=-.005, high=.005, size=self.model.nq).reshape(-1, 1)
        self.model.data.qvel = self.init_qvel + np.random.uniform(low=-.005, high=.005, size=self.model.nv).reshape(-1, 1)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()

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

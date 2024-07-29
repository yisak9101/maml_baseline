import random

import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides


class CheetahVelInter(MujocoEnv, Serializable):

    FILE = 'half_cheetah.xml'

    def __init__(self, *args, **kwargs):
        self.goals = []
        for i in range(100):
            prob = random.random()
            if prob >= 0.5:
                vel_train = random.uniform(0, 0.5)
            else:
                vel_train = random.uniform(3.0, 3.5)
            self.goals.append(vel_train)
        self.goal = self.sample_goal()

        super(CheetahVelInter, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def sample_goal(self):
        return random.choice(self.goals)
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flatten()[1:],
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def step(self, action):
        xposbefore = self.model.data.qpos[0][0]
        self.forward_dynamics(action)
        xposafter = self.model.data.qpos[0][0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self.goal)

        action = np.clip(action, *self.action_bounds)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        reward = forward_reward - ctrl_cost

        done = False
        next_obs = self.get_current_obs()
        return Step(next_obs, reward, done)

    @overrides
    def reset(self, init_state=None, **kwargs):
        self.goal = self.sample_goal()
        self.reset_mujoco(init_state)
        self.model.data.qpos = self.init_qpos + np.random.uniform(size=self.model.nq, low=-.1, high=.1)
        self.model.data.qvel = self.init_qvel + np.random.randn(self.model.nv) * .1
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)

    @overrides
    def log_diagnostics(self, paths, prefix=''):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular(prefix+'AverageForwardProgress', np.mean(progs))
        logger.record_tabular(prefix+'MaxForwardProgress', np.max(progs))
        logger.record_tabular(prefix+'MinForwardProgress', np.min(progs))
        logger.record_tabular(prefix+'StdForwardProgress', np.std(progs))

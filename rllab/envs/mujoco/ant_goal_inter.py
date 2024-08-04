import random

from .mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
import numpy as np

from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger


class AntGoalInter(MujocoEnv, Serializable):
    FILE = 'ant.xml'


    def __init__(self, *args, **kwargs):
        self.goals = []
        for i in range(150):
            prob = random.random()  # np.random.uniform()
            if prob < 4.0 / 15.0:
                r = random.random() ** 0.5  # [0, 1]
            else:
                r = (random.random() * 2.75 + 6.25) ** 0.5
            theta = random.random() * 2 * np.pi  # [0.0, 2pi]
            self.goals.append([r * np.cos(theta), r * np.sin(theta)])

        super(AntGoalInter, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        self.reset()

    def sample_goals(self, num_goals):
        return np.random.choice(self.goals, num_goals)

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)

    @overrides
    def reset(self, init_state=None, reset_args=None, **kwargs):
        if reset_args is not None:
            self.goal = reset_args
        elif self.goal is None:
            self.goal = self.sample_goals(1)[0]
        self.reset_mujoco(init_state)
        self.model.data.qpos = self.init_qpos + np.random.uniform(size=self.model.nq, low=-.1, high=.1)
        self.model.data.qvel = self.init_qvel + np.random.randn(self.model.nv) * .1
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        obs = self.get_current_obs()
        return obs

    def step(self, action):
        self.forward_dynamics(action)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self.goal)) # make it happy, not suicidal

        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.1 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
        survive_reward = 0.
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward

        done = False
        ob = self.get_current_obs()
        return Step(ob, float(reward), done)

    @overrides
    def log_diagnostics(self, paths, prefix=''):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular(prefix + 'AverageForwardProgress', np.mean(progs))
        logger.record_tabular(prefix + 'MaxForwardProgress', np.max(progs))
        logger.record_tabular(prefix + 'MinForwardProgress', np.min(progs))
        logger.record_tabular(prefix + 'StdForwardProgress', np.std(progs))

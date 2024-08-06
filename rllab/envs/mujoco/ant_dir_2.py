import random

from .mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
import numpy as np

from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger


class AntDir2(MujocoEnv, Serializable):
    FILE = 'ant.xml'

    def __init__(self, *args, **kwargs):
        self.goals = np.array([0.0, 0.5 * np.pi])
        self.goal = None
        super(AntDir2, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        self.reset()

    def sample_goals(self, num_goals):
        return random.Generator.choice(self.goals, num_goals)

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
        self.model.data.qpos = self.init_qpos + np.random.uniform(size=self.model.nq, low=-.1, high=.1).reshape(-1, 1)
        self.model.data.qvel = self.init_qvel + np.random.randn(self.model.nv).reshape(-1, 1) * .1
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        obs = self.get_current_obs()
        return obs

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self.goal), np.sin(self.goal))

        self.forward_dynamics(action)

        torso_xyz_after = np.array(self.get_body_com("torso"))

        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2] / self.dt), direct)

        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
        survive_reward = 1.
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        state = self.state_vector()
        notdone = np.isfinite(state).all() and 0.2 <= state[2] <= 1.0
        done = not notdone

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

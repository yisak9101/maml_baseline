import numpy as np

from rllab.envs.mujoco.cheetah_vel_inter import CheetahVelInter


class CheetahVelInterEval(CheetahVelInter):
    def __init__(self, *args, **kwargs):
        super(CheetahVelInterEval, self).__init__(*args, **kwargs)
        self.goals = [0.1, 0.25, 3.1, 3.25, 0.75, 1.25, 1.75, 2.25, 2.75]
        self.goal = self.sample_goal()

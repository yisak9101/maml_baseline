from rllab.envs.mujoco.ant_dir_4 import AntDir4
from rllab.misc.overrides import overrides
import numpy as np


class AntDir4Eval(AntDir4):
    def __init__(self, *args, **kwargs):
        super(AntDir4Eval, self).__init__(*args, **kwargs)
        self.goals =[0.25 * np.pi, 0.75 * np.pi, 1.25 * np.pi, 1.75 * np.pi]
        self.goal = self.sample_goal()

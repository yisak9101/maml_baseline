from rllab.envs.mujoco.ant_dir_2 import AntDir2
from rllab.misc.overrides import overrides
import numpy as np


class AntDir2Eval(AntDir2):

    def __init__(self, *args, **kwargs):
        super(AntDir2Eval, self).__init__(*args, **kwargs)
        self.goals = [3 * np.pi / 4, 7 * np.pi / 4, 1 * np.pi / 4]
        self.goal = self.sample_goals(1)[0]

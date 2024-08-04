import random

import numpy as np

from rllab.envs.mujoco.hopper_mass_inter import HopperMassInter
from rllab.misc.overrides import overrides


class HopperMassInterEval(HopperMassInter):
    def __init__(self, *args, **kwargs):
        super(HopperMassInterEval, self).__init__(*args, **kwargs)
        self.goals = [0.75, 1.25, 1.75, 2.25, 2.75, 0.1, 0.25, 3.1, 3.25]
        self.goal = self.sample_goals(1)[0]

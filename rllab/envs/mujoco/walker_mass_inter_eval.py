import random

import numpy as np

from rllab.envs.mujoco.walker_mass_inter import WalkerMassInter
from rllab.misc.overrides import overrides


class WalkerMassInterEval(WalkerMassInter):
    def __init__(self, *args, **kwargs):
        super(WalkerMassInterEval, self).__init__(*args, **kwargs)
        self.goals = [0.75, 1.25, 1.75, 2.25, 2.75]
        self.goal = self.sample_goals(1)[0]

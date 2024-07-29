import random

import numpy as np

from rllab.envs.mujoco.walker_mass_inter import WalkerMassInter
from rllab.misc.overrides import overrides


class WalkerMassInterEval(WalkerMassInter):
    @overrides
    def sample_mass_multiplier(self):
        return random.choice([0.75, 1.25, 1.75, 2.25, 2.75, 0.1, 0.25, 3.1, 3.25])

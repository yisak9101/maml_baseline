import numpy as np

from rllab.envs.mujoco.ant_goal_inter import AntGoalInter


class AntGoalInterEval(AntGoalInter):
    def __init__(self, *args, **kwargs):
        super(AntGoalInterEval, self).__init__(*args, **kwargs)
        self.goals = [[1.75, 0], [0, 1.75], [-1.75, 0], [0, -1.75]]
        self.goal = self.sample_goals(1)[0]
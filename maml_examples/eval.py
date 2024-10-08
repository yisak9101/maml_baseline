from sandbox.rocky.tf.algos.vpg import VPG
from sandbox.rocky.tf.policies.minimal_gauss_mlp_policy import GaussianMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
import tensorflow as tf

import csv
import numpy as np
import pickle

class Eval:
    def __init__(self, env, exp_name):
        self.env = env
        self.exp_name = exp_name

    def run(self, file_path):
        stub(globals())

        np.random.seed(2)

        env = self.env
        while 'goals' not in dir(env):
            env = env.wrapped_env
        goals = env.goals

        step_sizes = [0.1]
        initial_params_files = [file_path]


        all_avg_returns = []
        for step_i, initial_params_file in zip(range(len(step_sizes)), initial_params_files):
            avg_returns = []
            for goal_i, goal in zip(range(len(goals)), goals):

                n_itr = 4
                policy = GaussianMLPPolicy(  # random policy
                    name='policy',
                    env_spec=self.env.spec,
                    hidden_nonlinearity=tf.nn.relu,
                    hidden_sizes=(100, 100),
                )

                if initial_params_file is not None:
                    policy = None

                baseline = LinearFeatureBaseline(env_spec=self.env.spec)
                algo = VPG(
                    env=self.env,
                    policy=policy,
                    load_policy=initial_params_file,
                    baseline=baseline,
                    batch_size=8000,
                    max_path_length=200,
                    n_itr=n_itr,
                    reset_arg=goal,
                    optimizer_args={'init_learning_rate': step_sizes[step_i],
                                    'tf_optimizer_args': {'learning_rate': 0.01 * step_sizes[step_i]},
                                    'tf_optimizer_cls': tf.train.GradientDescentOptimizer}
                )

                run_experiment_lite(
                    algo.train(),
                    # Number of parallel workers for sampling
                    n_parallel=4,
                    # Only keep the snapshot parameters for the last iteration
                    snapshot_mode="all",
                    # Specifies the seed for the experiment. If this is not provided, a random seed
                    # will be used
                    seed=goal_i,
                    exp_prefix='test-maml',
                    exp_name=self.exp_name,
                    plot=True,
                )

                # get return from the experiment
                with open(f'data/local/test-maml/{self.exp_name}/progress.csv', 'r') as f:
                    reader = csv.reader(f, delimiter=',')
                    i = 0
                    row = None
                    returns = []
                    for row in reader:
                        i += 1
                        if i == 1:
                            ret_idx = row.index('AverageReturn')
                        else:
                            returns.append(float(row[ret_idx]))
                    avg_returns.append(returns)

            all_avg_returns.append(avg_returns)

            task_avg_returns = []
            for itr in range(len(all_avg_returns[step_i][0])):
                task_avg_returns.append([ret[itr] for ret in all_avg_returns[step_i]])

            results = {'task_avg_returns': task_avg_returns}

        for i in range(len(initial_params_files)):
            returns = []
            std_returns = []
            returns.append(np.mean([ret[itr] for ret in all_avg_returns[i]]))
            std_returns.append(np.std([ret[itr] for ret in all_avg_returns[i]]))
            print(initial_params_files[i])
            print(returns)
            print(std_returns)

        return returns[-1]
import matplotlib
matplotlib.use('Pdf')

import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import rllab.misc.logger as logger
import rllab.plotter as plotter
import tensorflow as tf
import time
import wandb
import json

from rllab.algos.base import RLAlgorithm
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
from sandbox.rocky.tf.spaces import Discrete
from rllab.sampler.stateful_pool import singleton_pool

class BatchMAMLPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods, with maml.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            scope=None,
            n_itr=500,
            start_itr=0,
            # Note that the number of trajectories for grad upate = batch_size
            # Defaults are 10 trajectories of length 500 for gradient update
            batch_size=100,
            max_path_length=500,
            meta_batch_size = 100,
            num_grad_updates=1,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            fixed_horizon=False,
            sampler_cls=None,
            sampler_args=None,
            force_batch_sampler=False,
            use_maml=True,
            load_policy=None,
            eval=None,
            env_name=None,
            seed=None,
            eval_interval_itr=None,
            kl_constraint=None,
            reward_scaling=None,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.  #
        :param max_path_length: Maximum length of a single rollout.
        :param meta_batch_size: Number of tasks sampled per meta-update
        :param num_grad_updates: Number of fast gradient updates
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.env = env
        self.policy = policy
        self.load_policy=load_policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        # batch_size is the number of trajectories for one fast grad update.
        # self.batch_size is the number of total transitions to collect.
        self.batch_size = batch_size * max_path_length * meta_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.meta_batch_size = meta_batch_size # number of tasks
        self.num_grad_updates = num_grad_updates # number of gradient steps during training

        if sampler_cls is None:
            if singleton_pool.n_parallel > 1:
                sampler_cls = BatchSampler
            else:
                sampler_cls = VectorizedSampler
        if sampler_args is None:
            sampler_args = dict()
        sampler_args['n_envs'] = self.meta_batch_size
        self.sampler = sampler_cls(self, **sampler_args)
        self.frames = 0
        self.eval_interval_itr = eval_interval_itr
        self.eval = eval
        if eval is not None:
            wandb.login(key="7316f79887c82500a01a529518f2af73d5520255")
            wandb.init(
                entity='mlic_academic',
                project='김정모_metaRL_baselines',
                group=env_name,
                name= f'maml-{env_name}-seed_{str(seed)}-klconstraint_{kl_constraint}-rewardscaling_{reward_scaling}'
            )
        self.reward_scaling = reward_scaling

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr, reset_args=None, log_prefix=''):
        # This obtains samples using self.policy, and calling policy.get_actions(obses)
        # return_dict specifies how the samples should be returned (dict separates samples
        # by task)
        paths = self.sampler.obtain_samples(itr, reset_args, return_dict=True, log_prefix=log_prefix)
        assert type(paths) == dict
        return paths

    def process_samples(self, itr, paths, prefix='', log=True, reward_scaling=1.0):
        return self.sampler.process_samples(itr, paths, prefix=prefix, log=log, reward_scaling=reward_scaling)

    def train(self):
        # TODO - make this a util
        flatten_list = lambda l: [item for sublist in l for item in sublist]
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)

        with tf.Session(config=config) as sess:
            # Code for loading a previous policy. Somewhat hacky because needs to be in sess.
            if self.load_policy is not None:
                import joblib
                self.policy = joblib.load(self.load_policy)['policy']
            self.init_opt()
            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = []
            for var in tf.global_variables():
                # note - this is hacky, may be better way to do this in newer TF.
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninit_vars.append(var)
            sess.run(tf.variables_initializer(uninit_vars))

            self.start_worker()
            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                with logger.prefix('itr #%d | ' % itr):
                    logger.log("Sampling set of tasks/goals for this meta-batch...")

                    env = self.env
                    while 'sample_goals' not in dir(env):
                        env = env.wrapped_env
                    learner_env_goals = env.sample_goals(self.meta_batch_size)

                    self.policy.switch_to_init_dist()  # Switch to pre-update policy

                    all_samples_data, all_paths, train_avg_returns = [], [], []
                    for step in range(self.num_grad_updates+1):
                        #if step > 0:
                        #    import pdb; pdb.set_trace() # test param_vals functions.
                        logger.log('** Step ' + str(step) + ' **')
                        logger.log("Obtaining samples...")
                        paths = self.obtain_samples(itr, reset_args=learner_env_goals, log_prefix=str(step))
                        for env in paths.values():
                            for episode in env:
                                self.frames += episode["rewards"].__len__()
                        all_paths.append(paths)
                        logger.log(f"Processing samples... (frames: {self.frames})")
                        samples_data = {}
                        for key in paths.keys():  # the keys are the tasks
                            # don't log because this will spam the consol with every task.
                            samples_data[key], train_avg_return = self.process_samples(itr, paths[key], log=False, reward_scaling=self.reward_scaling)
                            train_avg_returns.append(train_avg_return)
                        all_samples_data.append(samples_data)
                        # for logging purposes only
                        self.process_samples(itr, flatten_list(paths.values()), prefix=str(step), log=True)
                        logger.log("Logging diagnostics...")
                        self.log_diagnostics(flatten_list(paths.values()), prefix=str(step))
                        if step < self.num_grad_updates:
                            logger.log("Computing policy updates...")
                            self.policy.compute_updated_dists(samples_data)


                    logger.log("Optimizing policy...")
                    # This needs to take all samples_data so that it can construct graph for meta-optimization.
                    self.optimize_policy(itr, all_samples_data)
                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(itr, all_samples_data[-1])  # , **kwargs)
                    if self.store_paths:
                        params["paths"] = all_samples_data[-1]["paths"]
                    param_path = logger.save_itr_params(itr, params)
                    logger.save_itr_params(itr, params)
                    logger.log("Saved")
                    logger.record_tabular('Time', time.time() - start_time)
                    logger.record_tabular('ItrTime', time.time() - itr_start_time)

                    logger.dump_tabular(with_prefix=False)

                    if self.eval_interval_itr is not None and \
                            itr % self.eval_interval_itr == 0 and \
                            param_path is not None and \
                            self.eval is not None:
                        logger.log("Evaluating...")
                        test_avg_return = self.eval.run(param_path)
                        wandb_log_dict = {
                            "Eval/train_avg_return": np.array(train_avg_returns).mean(),
                            "Eval/test_avg_return": test_avg_return,
                        }
                        logger.log(f"timestep {self.frames}:{json.dumps(wandb_log_dict)}")
                        wandb.log(wandb_log_dict, step=self.frames)
        self.shutdown_worker()

    def log_diagnostics(self, paths, prefix):
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)

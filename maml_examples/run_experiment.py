from maml_examples.eval import Eval
from rllab.envs.mujoco.ant_dir_2 import AntDir2
from rllab.envs.mujoco.ant_dir_2_eval import AntDir2Eval
from rllab.envs.mujoco.ant_dir_4 import AntDir4
from rllab.envs.mujoco.ant_dir_4_eval import AntDir4Eval
from rllab.envs.mujoco.ant_goal_inter import AntGoalInter
from rllab.envs.mujoco.ant_goal_inter_eval import AntGoalInterEval
from rllab.envs.mujoco.cheetah_vel_inter import CheetahVelInter
from rllab.envs.mujoco.cheetah_vel_inter_eval import CheetahVelInterEval
from rllab.envs.mujoco.hopper_mass_inter import HopperMassInter
from rllab.envs.mujoco.hopper_mass_inter_eval import HopperMassInterEval
from rllab.envs.mujoco.walker_mass_inter import WalkerMassInter
from rllab.envs.mujoco.walker_mass_inter_eval import WalkerMassInterEval
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.minimal_gauss_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

import tensorflow as tf
import argparse


def main(env_name: str, seed: int):
    stub(globals())

    if env_name == 'ant-dir-2개':
        env = TfEnv(normalize(AntDir2()))
        eval_env = TfEnv(normalize(AntDir2Eval()))
    elif env_name == 'ant-dir-4개':
        env = TfEnv(normalize(AntDir4()))
        eval_env = TfEnv(normalize(AntDir4Eval()))
    elif env_name == 'ant-goal-inter':
        env = TfEnv(normalize(AntGoalInter()))
        eval_env = TfEnv(normalize(AntGoalInterEval()))
    elif env_name == 'cheetah-vel-inter':
        env = TfEnv(normalize(CheetahVelInter()))
        eval_env = TfEnv(normalize(CheetahVelInterEval()))
    elif env_name == 'hopper-mass-inter':
        env = TfEnv(normalize(HopperMassInter()))
        eval_env = TfEnv(normalize(HopperMassInterEval()))
    elif env_name == 'walker-mass-inter':
        env = TfEnv(normalize(WalkerMassInter()))
        eval_env = TfEnv(normalize(WalkerMassInterEval()))
    else:
        raise NameError

    exp_prefix = f"{env_name}-{seed}-maml"
    exp_name = f"{env_name}-{seed}-maml"

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_nonlinearity=tf.nn.relu,
        hidden_sizes=(100, 100),
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    eval = Eval(env=eval_env)
    max_path_length = 200

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=max_path_length * 100,  # number of trajs for grad update
        max_path_length=max_path_length,
        n_itr=int(1e10),
        use_maml=True,
        step_size=0.01,
        plot=False,
        eval=eval,
        env_name=env_name,
        seed=seed,
        eval_interval_itr=25,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix=exp_prefix,
        exp_name=exp_name,
        n_parallel=8,
        snapshot_mode="gap",
        snapshot_gap=25,
        sync_s3_pkl=True,
        seed=seed,
        mode="local",
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse command-line arguments.")
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    main(env_name=args.env, seed=args.seed)

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
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO

import tensorflow as tf
import argparse


def main(env_name: str, seed: int, kl_constraint: str, reward_scaling: float):
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

    exp_prefix = f"{env_name}-{seed}-{kl_constraint}-{reward_scaling}"
    exp_name = f"{env_name}-{seed}-{kl_constraint}-{reward_scaling}"

    policy = MAMLGaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        grad_step_size=0.1,
        hidden_nonlinearity=tf.nn.relu,
        hidden_sizes=(100,100),
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    eval = Eval(exp_name=exp_name ,env=eval_env)
    max_path_length = 200

    algo = MAMLTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=20, # number of trajs for grad update
        max_path_length=max_path_length,
        meta_batch_size=40,
        num_grad_updates=1,
        n_itr=int(1e10),
        use_maml=True,
        step_size=0.01,
        plot=False,
        eval=eval,
        env_name=env_name,
        seed=seed,
        eval_interval_itr=25,
        kl_constraint=kl_constraint,
        reward_scaling=reward_scaling
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
    parser.add_argument('--kl_constraint', type=str, required=True, help="kl constraint (min, max, mean)")
    parser.add_argument('--reward_scaling', type=float, required=True)
    args = parser.parse_args()

    main(env_name=args.env, seed=args.seed, kl_constraint=args.kl_constraint, reward_scaling=args.reward_scaling)

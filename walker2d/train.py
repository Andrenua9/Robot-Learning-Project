import argparse
from typing import Callable
import gymnasium as gym
import numpy as np
import os
from env.custom_walker2d import *
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.results_plotter import load_results, ts2xy

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: The initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining parte da 1.0 (inizio) e va a 0.0 (fine)
        return progress_remaining * initial_value
    return func


CONFIG_A = dict(policy='MlpPolicy',
                learning_rate=linear_schedule(3e-4),
                gamma=0.99,
                n_steps=2048,
                batch_size=512,
                n_epochs=5,
                ent_coef=0.003,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])))

CONFIG_B = dict(policy='MlpPolicy',
                learning_rate=5e-5,
                gamma=0.99,
                n_steps=512,
                batch_size=64,
                n_epochs=10,
                ent_coef=0.003,
                vf_coef=0.87,
                max_grad_norm=1)

CONFIG_C = dict(policy='MlpPolicy',
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99)

CONFIG_D = dict(policy='MlpPolicy',
                learning_rate=3e-4,
                gamma=0.99,
                n_steps=2048,
                batch_size=512,
                n_epochs=10,
                ent_coef=0.002,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])))

CONFIG_E = dict(policy='MlpPolicy',
                learning_rate=3e-4,
                gamma=0.99,
                n_steps=2048,
                batch_size=128,
                n_epochs=10,
                ent_coef=0.001)

CONFIGS = dict(a=CONFIG_A, b=CONFIG_B, c=CONFIG_C, d=CONFIG_D, e=CONFIG_E)

def get_net_arch(size):
    """Return the architecture of the network based on the size chosen"""
    if size == 'small':
        return dict(pi=[64, 64], vf=[64, 64])
    elif size == 'medium':
        return dict(pi=[256, 256], vf=[256, 256])
    elif size == 'deep':
        return dict(pi=[256, 256, 256], vf=[256, 256, 256])
    else:
        raise ValueError("Choose from 'small', 'medium', or 'deep'")

def train_model(env_name, args, log_dir=None, verbose=0):
    net_arch=get_net_arch(args.net_arch)
    policy_kwargs = dict(net_arch=net_arch)
    if args.udr_range > 0.0 and 'source' in env_name:
        vec_env = make_vec_env(env_name, n_envs=8, monitor_dir=log_dir, 
                                env_kwargs={"udr_range": args.udr_range, "symmetric": args.symmetric}, seed=args.seed)
        print(f"Using UDR with range ±{args.udr_range*100}% with {args.net_arch} architecture for training.")
    else:
        vec_env = make_vec_env(env_name, n_envs=8, monitor_dir=log_dir, seed=args.seed)

    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    if args.config:
        print(f"Using predefined configuration '{args.config}' for training.")
        model = PPO(**CONFIGS[args.config], env=vec_env, verbose=verbose, seed=args.seed)
    else:
        print(f"Using custom configuration with {args.net_arch} architecture for training.")
        model=PPO(
            'MlpPolicy',
            vec_env,
            learning_rate=args.lr,
            gamma=args.gamma,
            batch_size=64,
            n_steps=256,
            clip_range=0.2,
            n_epochs=10,
            ent_coef=0,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=args.seed
            )
    model.learn(total_timesteps=args.timesteps)
    os.makedirs("stats", exist_ok=True)
    udr_tag = ""
    if args.udr_range > 0.0:
        udr_tag = f"_udr{int(args.udr_range*100)}"
        if args.symmetric:
            udr_tag += "sym"
    if args.config:
        stats_path = f"./stats/vec_normalize_{args.env}_{args.config}{udr_tag}_seed{args.seed}.pkl"
    else:
        stats_path = f"./stats/vec_normalize_{args.env}_{args.net_arch}{udr_tag}_seed{args.seed}.pkl"
    vec_env.save(stats_path)
    vec_env.close()
    return model

def evaluate_model(model, env_name, args, n_episodes=100, render_mode=None):
    env_test = DummyVecEnv([lambda: Monitor(gym.make(env_name, render_mode=render_mode))])
    udr_tag = ""
    if args.udr_range > 0.0:
        udr_tag = f"_udr{int(args.udr_range*100)}"
        if args.symmetric:
            udr_tag += "sym"
    if args.config:
        stats_path = f"./stats/vec_normalize_{args.env}_{args.config}{udr_tag}_seed{args.seed}.pkl"
    else:
        stats_path = f"./stats/vec_normalize_{args.env}_{args.net_arch}{udr_tag}_seed{args.seed}.pkl"
    env_test = VecNormalize.load(stats_path, env_test)
    env_test.training = False
    env_test.norm_reward = False
    print(f'Masses for {env_name}: {env_test.env_method("get_parameters")[0]}')
    mean_reward, std_reward = evaluate_policy(model, env_test, n_eval_episodes=n_episodes, render=(render_mode is not None))
    env_test.close()
    return mean_reward, std_reward

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")

def plot_results(log_folder, title="Learning Curve", algo=""):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    :param algo: (str) the algorithm name for filename
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    
    # Save plot to plots directory
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/learning_curve_{algo}.png")
    #plt.show()
    plt.close(fig)

def main(args):
    if args.seed>0.0:
        set_random_seed(args.seed)

    if args.test:
        if args.config:
            model_path = f"./models/ppo_walker2d_{args.env}_{args.config}"
        else:
            model_path = f"./models/ppo_walker2d_{args.env}_{args.net_arch}"
        udr_tag = ""
        if args.udr_range > 0.0:
            udr_tag = f"_udr{int(args.udr_range*100)}"
            if args.symmetric:
                udr_tag += "sym"
        model_path += f"{udr_tag}_seed{args.seed}"
        print(f"Evaluating model trained on {args.env} environment ({model_path})...")
        model = PPO.load(model_path)
        render_mode = 'human' if args.render_test else None

        if args.env == 'source':
            print(f"\nTesting on source environment ({args.env}->source)...")
            mean_reward, std_reward = evaluate_model(model, 'CustomWalker2d-source-v0', args, args.n_episodes, render_mode=render_mode)
            print(f"Source Env: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        print(f"\nTesting on target environment ({args.env}->target)...")
        mean_reward, std_reward = evaluate_model(model, 'CustomWalker2d-target-v0', args, args.n_episodes, render_mode=render_mode)
        print(f"Target Env: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    else:
        if args.config:
            log_dir = f"./logs/walker2d_{args.env}_{args.config}"
        else:
            log_dir = f"./logs/walker2d_{args.env}_{args.net_arch}"
        udr_tag = ""
        if args.udr_range > 0.0:
            udr_tag = f"_udr{int(args.udr_range*100)}"
            if args.symmetric:
                udr_tag += "sym"
        log_dir += f"{udr_tag}_seed{args.seed}"
        os.makedirs(log_dir, exist_ok=True)
        
        env_name = f'CustomWalker2d-{args.env}-v0'
        print(f"Training on {env_name} environment...")
        model = train_model(env_name, args, log_dir=log_dir, verbose=1)

        if args.config:
            model_path = f"./models/ppo_walker2d_{args.env}_{args.config}"
            algo_name = f"ppo_{args.env}_{args.config}"
        else:
            model_path = f"./models/ppo_walker2d_{args.env}_{args.net_arch}"
            algo_name = f"ppo_{args.env}_{args.net_arch}"

        udr_tag = ""
        if args.udr_range > 0.0:
            udr_tag = f"_udr{int(args.udr_range*100)}"
            if args.symmetric:
                udr_tag += "sym"
        model_path += f"{udr_tag}_seed{args.seed}"
        algo_name += f"{udr_tag}_seed{args.seed}"
        model.save(model_path)
        print(f"Model saved to {model_path}")
        plot_results(log_dir, algo=algo_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test PPO on CustomWalker2d')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--test', action='store_true', help='Test mode (otherwise train)')
    parser.add_argument('--env', type=str, choices=['source', 'target'], default='source',
                        help='Environment to train on (source or target)')
    parser.add_argument('--n_episodes', type=int, default=100,
                        help='Number of episodes for evaluation')
    parser.add_argument('--timesteps', type=int, default=3_000_000,
                        help='Total timesteps for training')
    parser.add_argument('--render_test', action='store_true',
                        help='Render the environment during testing')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for training')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor gamma')
    parser.add_argument('--udr_range', type=float, default=0.0, help='UDR range (0.0=disabled, 0.25=±25%, etc.)')
    parser.add_argument('--net_arch', type=str, default='small', help='Network architecture (small, medium, large)')
    parser.add_argument('--config', type=str, choices=['a', 'b', 'c', 'd', 'e'], help='Predefined configuration to use (a, b, c, d, or e)')
    parser.add_argument('--symmetric', action='store_true', default=False, help='Use symmetric mass randomization (only for UDR)')
    args = parser.parse_args()
    main(args)
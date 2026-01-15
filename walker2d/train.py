import argparse
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
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy

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
                                env_kwargs={"udr_range": args.udr_range}, seed=args.seed)
        print(f"Using UDR with range ±{args.udr_range*100}% with {args.net_arch} architecture for training.")
    else:
        vec_env = make_vec_env(env_name, n_envs=8, monitor_dir=log_dir, seed=args.seed)

    model=PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=256,
        n_steps=2048,
        clip_range=0.2,
        n_epochs=10,
        ent_coef=0.001,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        seed=args.seed
        )
    model.learn(total_timesteps=args.timesteps)
    vec_env.close()
    return model

def evaluate_model(model, env_name, n_episodes=50, render_mode=None):
    env_test = gym.make(env_name, render_mode=render_mode)
    env_test = Monitor(env_test)
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
    plt.show()

def main(args):
    if args.seed>0.0:
        set_random_seed(args.seed)

    if args.test:
        model_path = f"./models/ppo_walker2d_{args.env}_{args.net_arch}"
        if args.udr_range > 0.0:
            model_path += f"_udr_{int(args.udr_range*100)}"
        print(f"Evaluating model trained on {args.env} environment ({model_path})...")
        model = PPO.load(model_path)
        render_mode = 'human' if args.render_test else None

        if args.env == 'source':
            print(f"\nTesting on source environment ({args.env}->source)...")
            mean_reward, std_reward = evaluate_model(model, 'CustomWalker2d-source-v0', args.n_episodes, render_mode=render_mode)
            print(f"Source Env: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        print(f"\nTesting on target environment ({args.env}->target)...")
        mean_reward, std_reward = evaluate_model(model, 'CustomWalker2d-target-v0', args.n_episodes, render_mode=render_mode)
        print(f"Target Env: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    else:
        log_dir = f"./logs/walker2d_{args.env}_{args.net_arch}"
        os.makedirs(log_dir, exist_ok=True)
        env_name = f'CustomWalker2d-{args.env}-v0'
        print(f"Training on {env_name} environment...")
        model = train_model(env_name, args, log_dir=log_dir, verbose=1)

        model_path = f"./models/ppo_walker2d_{args.env}_{args.net_arch}"
        algo_name = f"ppo_{args.env}_{args.net_arch}"

        if args.udr_range > 0.0:
            model_path += f"_udr_{int(args.udr_range*100)}"
            algo_name += f"_udr_{int(args.udr_range*100)}"
        model.save(model_path)
        print(f"Model saved to {model_path}")
        plot_results(log_dir, algo=algo_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test PPO on CustomHopper')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--test', action='store_true', help='Test mode (otherwise train)')
    parser.add_argument('--env', type=str, choices=['source', 'target'], default='source',
                        help='Environment to train on (source or target)')
    parser.add_argument('--n_episodes', type=int, default=50,
                        help='Number of episodes for evaluation')
    parser.add_argument('--timesteps', type=int, default=1_500_000,
                        help='Total timesteps for training')
    parser.add_argument('--render_test', action='store_true',
                        help='Render the environment during testing')
    parser.add_argument('--optimize', action='store_true',
                        help='Run different hyperparameters and save the best model')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for training')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor gamma')
    parser.add_argument('--udr_range', type=float, default=0.0, help='UDR range (0.0=disabled, 0.25=±25%, etc.)')
    parser.add_argument('--net_arch', type=str, default='medium', help='Network architecture (small, medium, large)')

    args = parser.parse_args()
    main(args)
"""Sample script for training a control policy on the Hopper environment

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between TRPO, PPO, and SAC.
"""
import argparse
import gymnasium as gym
import numpy as np
import os
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy

def train_model(env_name, args, log_dir=None, verbose=0):
    """
    Train a PPO model on the specified environment
    
    :param env_name: (str) Environment name (e.g., 'CustomHopper-source-v0')
    :param timesteps: (int) Total timesteps for training
    :param lr: (float) Learning rate
    :param gamma: (float) Discount factor
    :param log_dir: (str) Directory for logging (optional)
    :param verbose: (int) Verbosity level
    :param udr_range: (float) UDR range (0.0=no UDR, 0.25=±25%, etc.)
    :param seed: (int) Random seed for reproducibility
    :return: (PPO) Trained model
    """
    # Create environment with optional UDR
    if args.udr_range > 0.0 and 'source' in env_name:
        vec_env = make_vec_env(env_name, n_envs=8, monitor_dir=log_dir, 
                               env_kwargs={"udr_range": args.udr_range}, seed=args.seed)
        print(f"Using UDR with range ±{args.udr_range*100}% for training.")
    else:
        vec_env = make_vec_env(env_name, n_envs=8, monitor_dir=log_dir, seed=args.seed)
    
    model = PPO(
        'MlpPolicy',
        vec_env,
        n_steps=2048,
        batch_size=64,
        learning_rate=args.lr,
        gamma=args.gamma,
        n_epochs=10,
        clip_range=0.2,
        verbose=verbose,
            ent_coef=0.0,
        seed=10
    )
    model.learn(total_timesteps=args.timesteps)
    vec_env.close()
    return model

def evaluate_model(model, env_name, args, n_episodes=50, render_mode=None):
    """
    Evaluate a model trained on the specified environment
    
    :param model: (PPO) Model to evaluate
    :param env_name: (str) Environment name (e.g., 'CustomHopper-source-v0')
    :param n_episodes: (int) Number of episodes for evaluation
    :param render_mode: (str) Render mode ('human' or None)
    :return: (tuple) Mean reward and standard deviation
    """
    env_test = DummyVecEnv([lambda: Monitor(gym.make(env_name, render_mode=render_mode))])
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
    # Set random seed
    if args.seed > 0:
        set_random_seed(args.seed)

    if args.test:
        # Test mode: load model and evaluate on both environments
        model_path = f"./models/ppo_hopper_{args.env}"
        udr_tag = ""
        if args.udr_range > 0.0:
            udr_tag = f"_udr{int(args.udr_range*100)}"
        model_path += f"{udr_tag}_seed{args.seed}"
        print(f"Evaluating model trained on {args.env} environment ({model_path})...")
        model = PPO.load(model_path)
        render_mode = 'human' if args.render_test else None

        # Test on source env
        if args.env == 'source':
            print(f"\nTesting on source environment ({args.env}->source)...")
            mean_reward, std_reward = evaluate_model(model, 'CustomHopper-source-v0', args, args.n_episodes, render_mode)
            print(f"Source Env: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        # Test on target env
        print(f"\nTesting on target environment ({args.env}->target)...")
        mean_reward, std_reward = evaluate_model(model, 'CustomHopper-target-v0', args, args.n_episodes, render_mode)
        print(f"Target Env: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    else:
        # Training mode
        # Create log directory
        log_dir = f"./logs/hopper_{args.env}"
        udr_tag = ""
        if args.udr_range > 0.0:
            udr_tag = f"_udr{int(args.udr_range*100)}"
        log_dir += f"{udr_tag}_seed{args.seed}"
        os.makedirs(log_dir, exist_ok=True)

        print(f"Training on {args.env} environment...")
        # Train on specified env
        train_env_name = f'CustomHopper-{args.env}-v0'
        model = train_model(train_env_name, args, log_dir=log_dir, verbose=1)
        
        model_path = f"./models/ppo_hopper_{args.env}"
        algo_name = f"ppo_{args.env}"
        
        udr_tag = ""
        if args.udr_range > 0.0:
            udr_tag = f"_udr{int(args.udr_range*100)}"
        model_path += f"{udr_tag}_seed{args.seed}"
        algo_name += f"{udr_tag}_seed{args.seed}"
        model.save(model_path)
        print(f"Model saved to {model_path}")
        plot_results(log_dir, algo=algo_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test PPO on CustomHopper')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--test', action='store_true', help='Test mode (otherwise train)')
    parser.add_argument('--env', type=str, choices=['source', 'target'], default='source',
                        help='Environment to train on (source or target)')
    parser.add_argument('--n_episodes', type=int, default=50,
                        help='Number of episodes for evaluation')
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                        help='Total timesteps for training')
    parser.add_argument('--render_test', action='store_true',
                        help='Render the environment during testing')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for training')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor gamma')
    parser.add_argument('--udr_range', type=float, default=0.0, help='UDR range (0.0=disabled, 0.25=±25%, etc.)')
    
    args = parser.parse_args()
    main(args)
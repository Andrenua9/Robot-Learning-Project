"""Implementation of Active Domain Randomization (ADR) for robust policy training.

ADR gradually increases the domain randomization range during training based on the
agent's performance on the target domain. This helps learn progressively more robust policies.
"""
import argparse
import json
import gymnasium as gym
import numpy as np
import os
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from collections import deque
from pathlib import Path


class ADRTrainer:
    """Active Domain Randomization trainer.
    
    Starts with low domain randomization and progressively increases it based on
    the agent's performance on the target domain.
    """
    
    def __init__(
        self,
        initial_udr_range=0.0,
        max_udr_range=0.3,
        udr_step=0.01,
        performance_buffer_size=10,
        performance_lower_bound=150.0,
        performance_upper_bound=350.0,
        eval_frequency=50_000,
        eval_episodes=10,
        seed=0,
        verbose=1
    ):
        """
        :param initial_udr_range: Starting domain randomization range
        :param max_udr_range: Maximum domain randomization range
        :param udr_step: How much to increase/decrease UDR range
        :param performance_buffer_size: Size of the performance buffer for boundary evaluation
        :param performance_lower_bound: Lower bound for performance to trigger narrowing
        :param performance_upper_bound: Upper bound for performance to trigger widening
        :param eval_frequency: Evaluate every N timesteps
        :param eval_episodes: Number of episodes for evaluation
        :param seed: Random seed
        :param verbose: Verbosity level
        """
        self.initial_udr_range = initial_udr_range
        self.current_udr_range = initial_udr_range
        self.max_udr_range = max_udr_range
        self.udr_step = udr_step
        self.performance_buffer_size = performance_buffer_size
        self.performance_lower_bound = performance_lower_bound
        self.performance_upper_bound = performance_upper_bound
        self.eval_frequency = eval_frequency
        self.eval_episodes = eval_episodes
        self.seed = seed
        self.verbose = verbose
        
        self.total_timesteps = 0
        self.adr_history = []
        self.performance_buffer = deque(maxlen=performance_buffer_size)
        
    def update_difficulty(self):
        """Update domain randomization range based on performance buffer."""
        if len(self.performance_buffer) < self.performance_buffer_size:
            return False  # Not enough data to make a decision

        avg_performance = np.mean(self.performance_buffer)
        
        if self.verbose:
            print(f"ADR: Avg boundary performance: {avg_performance:.2f} | "
                  f"Bounds: [{self.performance_lower_bound}, {self.performance_upper_bound}]")

        old_range = self.current_udr_range
        
        if avg_performance > self.performance_upper_bound:
            # Widen the range
            self.current_udr_range = min(self.current_udr_range + self.udr_step, self.max_udr_range)
            if self.verbose and self.current_udr_range > old_range:
                print(f"📈 ADR: Performance high, widening range to ±{self.current_udr_range*100:.1f}%")
            return self.current_udr_range > old_range
            
        elif avg_performance < self.performance_lower_bound:
            # Narrow the range
            self.current_udr_range = max(self.current_udr_range - self.udr_step, 0.0)
            if self.verbose and self.current_udr_range < old_range:
                print(f"📉 ADR: Performance low, narrowing range to ±{self.current_udr_range*100:.1f}%")
            return self.current_udr_range < old_range
            
        return False

    def evaluate_on_boundaries(self, model, n_episodes):
        """Evaluate the policy at the boundaries of the current UDR range."""
        # Positive boundary
        pos_boundary_reward, _ = self.evaluate_on_target(
            model, n_episodes=n_episodes, udr_range=self.current_udr_range
        )
        # Negative boundary (assuming symmetric randomization)
        neg_boundary_reward, _ = self.evaluate_on_target(
            model, n_episodes=n_episodes, udr_range=-self.current_udr_range
        )
        
        avg_boundary_reward = (pos_boundary_reward + neg_boundary_reward) / 2.0
        self.performance_buffer.append(avg_boundary_reward)
        
        if self.verbose:
            print(f"Boundary Eval: Avg Reward: {avg_boundary_reward:.2f} | "
                  f"Buffer: {list(self.performance_buffer)}")

    def evaluate_on_target(self, model, n_episodes=10, udr_range=None):
        """Evaluate policy on a specified domain.
        
        :param model: PPO model to evaluate
        :param n_episodes: Number of evaluation episodes
        :param udr_range: UDR range for evaluation. If None, uses the nominal target env.
                         If a float, creates a source env with that randomization.
        :return: Mean reward and std reward
        """
        if udr_range is not None:
            # Evaluate on a randomized environment
            env_target = gym.make('CustomHopper-source-v0', udr_range=udr_range)
        else:
            # Evaluate on the nominal, non-randomized target environment
            env_target = gym.make('CustomHopper-target-v0')
        
        env_target = Monitor(env_target)
        mean_reward, std_reward = evaluate_policy(
            model, env_target, n_eval_episodes=n_episodes, deterministic=False
        )
        env_target.close()
        return mean_reward, std_reward
    
    def train(
        self,
        total_timesteps=1_000_000,
        log_dir=None,
        save_dir="./models/adr"
    ):
        """Train policy with active domain randomization.
        
        :param total_timesteps: Total training timesteps
        :param log_dir: Directory for training logs
        :param save_dir: Directory to save models
        :return: Trained model and ADR history
        """
        os.makedirs(save_dir, exist_ok=True)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Create source environment with initial UDR
        vec_env = make_vec_env(
            'CustomHopper-source-v0',
            n_envs=8,
            monitor_dir=log_dir,
            env_kwargs={"udr_range": self.current_udr_range},
            seed=self.seed
        )
        
        model = PPO('MlpPolicy', vec_env, verbose=0, seed=self.seed)
        
        if self.verbose:
            print(f"Starting ADR training: {total_timesteps} timesteps")
            print(f"Initial UDR range: ±{self.initial_udr_range*100:.1f}%")
            print(f"Max UDR range: ±{self.max_udr_range*100:.1f}%")
            print(f"Eval frequency: {self.eval_frequency} timesteps\n")
        
        remaining_timesteps = total_timesteps
        
        while remaining_timesteps > 0:
            # Train for eval_frequency timesteps
            train_steps = min(self.eval_frequency, remaining_timesteps)
            model.learn(total_timesteps=train_steps, reset_num_timesteps=False)
            
            self.total_timesteps += train_steps
            remaining_timesteps -= train_steps
            
            # Evaluate on boundaries and update performance buffer
            self.evaluate_on_boundaries(model, n_episodes=self.eval_episodes)
            
            # Evaluate on the nominal target for logging
            target_reward, target_std = self.evaluate_on_target(
                model, n_episodes=self.eval_episodes
            )
            
            # Log ADR history
            history_entry = {
                "total_timesteps": self.total_timesteps,
                "udr_range": self.current_udr_range,
                "target_reward": float(target_reward),
                "target_std": float(target_std),
                "avg_boundary_reward": np.mean(self.performance_buffer) if self.performance_buffer else 0,
            }
            self.adr_history.append(history_entry)
            
            if self.verbose:
                print(f"Timestep {self.total_timesteps:,} | "
                      f"UDR ±{self.current_udr_range*100:5.1f}% | "
                      f"Target Reward: {target_reward:7.2f} ± {target_std:6.2f}")
            
            # Update difficulty based on boundary performance
            difficulty_changed = self.update_difficulty()
            
            if difficulty_changed:
                # Recreate environment with the new randomization range
                vec_env.close()
                vec_env = make_vec_env(
                    'CustomHopper-source-v0',
                    n_envs=8,
                    monitor_dir=log_dir,
                    env_kwargs={"udr_range": self.current_udr_range},
                    seed=self.seed
                )
                model.set_env(vec_env)
                # Reset buffer after a difficulty change to gather fresh data
                self.performance_buffer.clear()
        
        vec_env.close()
        
        # Save final model and history
        model_path = os.path.join(save_dir, "adr_model")
        model.save(model_path)
        
        history_path = os.path.join(save_dir, "adr_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.adr_history, f, indent=2)
        
        if self.verbose:
            print(f"\n✅ Training complete!")
            print(f"Model saved to: {model_path}")
            print(f"History saved to: {history_path}")
        
        return model, self.adr_history
    
    def plot_adr_progress(self, save_path="./plots/adr_progress.png"):
        """Plot ADR progress over training."""
        import matplotlib.pyplot as plt
        
        if not self.adr_history:
            print("No ADR history to plot")
            return
        
        timesteps = [h["total_timesteps"] for h in self.adr_history]
        udr_ranges = [h["udr_range"] * 100 for h in self.adr_history]
        target_rewards = [h["target_reward"] for h in self.adr_history]
        boundary_rewards = [h["avg_boundary_reward"] for h in self.adr_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot UDR range
        ax1.plot(timesteps, udr_ranges, 'b-o', linewidth=2, markersize=4, label='UDR Range')
        ax1.set_ylabel("UDR Range (%)")
        ax1.set_title("Active Domain Randomization: Difficulty and Performance")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot rewards on a secondary y-axis
        ax1b = ax1.twinx()
        ax1b.plot(timesteps, target_rewards, 'g-s', linewidth=2, markersize=4, label='Target Reward')
        ax1b.plot(timesteps, boundary_rewards, 'm-^', linewidth=2, markersize=4, label='Avg Boundary Reward')
        ax1b.set_ylabel("Reward")
        ax1b.legend(loc='upper right')
        
        # Plot target domain reward vs performance bounds
        ax2.plot(timesteps, boundary_rewards, 'm-^', linewidth=2, markersize=4, label='Avg Boundary Reward')
        ax2.axhline(y=self.performance_lower_bound, color='r', linestyle='--', label=f'Lower Bound ({self.performance_lower_bound})')
        ax2.axhline(y=self.performance_upper_bound, color='g', linestyle='--', label=f'Upper Bound ({self.performance_upper_bound})')
        ax2.set_xlabel("Timesteps")
        ax2.set_ylabel("Average Boundary Reward")
        ax2.set_title("Boundary Performance vs. Dynamic Thresholds")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100)
        print(f"ADR progress plot saved to: {save_path}")
        plt.show()


def main(args):
    if args.seed > 0:
        set_random_seed(args.seed)
    
    # Create ADR trainer
    trainer = ADRTrainer(
        initial_udr_range=args.initial_udr_range,
        max_udr_range=args.max_udr_range,
        udr_step=args.udr_step,
        performance_buffer_size=args.performance_buffer_size,
        performance_lower_bound=args.performance_lower_bound,
        performance_upper_bound=args.performance_upper_bound,
        eval_frequency=args.eval_frequency,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        verbose=1
    )
    
    # Train with ADR
    log_dir = "./logs/adr_train/"
    model, history = trainer.train(
        total_timesteps=args.timesteps,
        log_dir=log_dir,
        save_dir="./models/adr"
    )
    
    # Plot results
    if args.plot:
        trainer.plot_adr_progress()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train with Active Domain Randomization')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                        help='Total timesteps for training')
    parser.add_argument('--initial_udr_range', type=float, default=0.0,
                        help='Initial UDR range (e.g., 0.05 = ±5%)')
    parser.add_argument('--max_udr_range', type=float, default=0.3,
                        help='Maximum UDR range (e.g., 0.3 = ±30%)')
    parser.add_argument('--udr_step', type=float, default=0.02,
                        help='How much to increase/decrease UDR range (e.g., 0.02 = ±2%)')
    parser.add_argument('--performance_buffer_size', type=int, default=10,
                        help='Size of the performance buffer for boundary evaluation')
    parser.add_argument('--performance_lower_bound', type=float, default=150.0,
                        help='Lower bound for performance to trigger narrowing')
    parser.add_argument('--performance_upper_bound', type=float, default=350.0,
                        help='Upper bound for performance to trigger widening')
    parser.add_argument('--eval_frequency', type=int, default=50_000,
                        help='Evaluate every N timesteps')
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of episodes for evaluation')
    parser.add_argument('--plot', action='store_true',
                        help='Plot ADR progress after training')
    
    args = parser.parse_args()
    main(args)

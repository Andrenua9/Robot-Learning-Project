'''
ADR implementation

Script to dynamically train a hopper based on his performance.

'''
import gymnasium as gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy



initial_udr_range=0.0,
performance_lower_bound=150.0, #mean reward could be into this interval, if overcame or is near to upper bound we update the range
performance_upper_bound=350.0,
check_frequency = 30000 #defines how many timesteps should pass between performance evaluations

class HopperADR(gym.Wrapper):
    

    def __init__(self,env,range): 
        super.__init__(env)

        self.masses = self.unwrapped.get_parameters()
        self.range = range

    def reset(self, **kwargs):
        return super.resest(**kwargs)
    
    def update_udr_range(self,new_range):
        self.update_udr_range = new_range

        

def train_agent(env, timesteps):
     model = PPO("MlpPolicy",env,learning_rate=0.002, verbose = 0, gamma=0.99) #hyperameters
     model.learn(total_timesteps=timesteps) #return a trained model
     return model

def test_agent(model, env, env_name="Environment"):
    print(f"--- Testing on {env_name} ---")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50, deterministic=True)
        
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print("-" * 30)
    return mean_reward


def main(args):
    pass
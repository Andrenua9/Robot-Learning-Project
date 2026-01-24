'''
ADR implementation

Script to dynamically train a hopper based on his performance.

'''
import gymnasium as gym
import numpy as np
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback


#it was necessary to implment this wrapper class in order to interact with the environment (in particular we use it to update the masses)
class HopperADR(gym.Wrapper):
    
    def __init__(self,env,range): 
        super().__init__(env)

        self.masses = self.unwrapped.get_parameters()
        self.range = range

# start a new episode, initialization
    def reset(self, seed = None, options = None):
        new_masses = np.array(self.masses).copy()
        low = new_masses[1:] * (1.0 - self.range)
        high = new_masses[1:] * (1.0 + self.range)
        new_masses = self.np_random.uniform(low=low, high=high)
        print(f"Nuove masse: {new_masses}")
        self.unwrapped.set_parameters(new_masses)
        return self.env.reset(seed = seed, options = options)
    

#we use callbacks to access internal state of the RL model during training
class HopperADRCallback(BaseCallback):
    def __init__(self, adr_wrapper, check_freq, up_thr, low_thr, step):
        super().__init__()
        self.adr_wrapper = adr_wrapper
        self.check_freq = check_freq
        self.up_thr = up_thr
        self.low_thr = low_thr
        self.step = step

    def _on_step(self) -> bool:

        if self.n_calls % self.check_freq == 0:

            mean_reward, _ = evaluate_policy(
                self.model,
                self.adr_wrapper,
                n_eval_episodes=5,
                deterministic=True
            )

            print(f"[ADR] Performance: {mean_reward:.2f} | Range = {self.adr_wrapper.range:.3f}")

            if mean_reward > self.up_thr:
                self.adr_wrapper.range += self.step
                print(f"[ADR] Range aumentato {self.adr_wrapper.range:.3f}")

            elif mean_reward < self.low_thr:
                self.adr_wrapper.range = max(0.0, self.adr_wrapper.range - self.step)
                print(f"[ADR] Range diminuito {self.adr_wrapper.range:.3f}")

        return True

    
    


def train_agent(env, timesteps, adr_callback):
     model = PPO("MlpPolicy",env,learning_rate=0.0003, verbose = 0, gamma=0.99) #hyperameters
     model.learn(total_timesteps=timesteps, callback=adr_callback) #return a trained model
     return model

def test_agent(model, env):
    print(f"--- Testing  ---")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50, deterministic=True)
        
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print("-" * 30)
    return mean_reward


def main():
    #parameters:
    initial_udr_range=0.0
    step_size = 0.05 #how much the range increases 
    performance_lower_bound=150.0 #mean reward could be into this interval, if overcame or is near to upper bound we update the range
    performance_upper_bound=350.0
    check_frequency = 30000 #defines how many timesteps should pass between performance evaluations

    base_env = gym.make("CustomHopper-source-v0")
    adr_env = HopperADR(env=base_env, range=initial_udr_range)


    adr_callback = HopperADRCallback(adr_wrapper=adr_env, check_freq=check_frequency, up_thr=performance_upper_bound, low_thr=performance_lower_bound, step=step_size)

    #Training and testing

    print("Training starts...")
    model = train_agent(env=base_env, timesteps=1000000, adr_callback=adr_callback)
    model.save("Hopper ADR model")

    test_agent(model, "CustomHopper-source-v0")
    test_agent(model, "CustomHopper-target-v0")

if __name__ == "__main__":
    main()


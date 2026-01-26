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
import matplotlib.pyplot as plt

#it was necessary to implment this wrapper class in order to interact with the environment (in particular we use it to update the masses)
class HopperADR(gym.Wrapper):
    
    def __init__(self,env,initial_udr_range): 
        super().__init__(env)

        if initial_udr_range > 1.0:
            raise ValueError("initial_udr_range must be <= 1")

        self.nominal_masses = self.unwrapped.get_parameters()
        self.low = self.nominal_masses[1:] * (1.0 - initial_udr_range)
        self.high = self.nominal_masses[1:] * (1.0 + initial_udr_range)
        self.evaluation_mode = False

# start a new episode, initialization
    def reset(self, seed=None, options=None):
        if not self.evaluation_mode:
            masses = self.np_random.uniform(self.low, self.high)
            print(f"Nuove masse: {masses}")
            self.unwrapped.set_parameters(masses)
        
        return self.env.reset(seed=seed, options=options)
    

#we use callbacks to access internal state of the RL model during training
class HopperADRCallback(BaseCallback):
    def __init__(self, adr_wrapper, check_freq, up_thr, low_thr, step):
        super().__init__()
        self.adr_wrapper = adr_wrapper
        self.check_freq = check_freq
        self.up_thr = up_thr
        self.low_thr = low_thr
        self.step = step

        self.timesteps = []
        self.mean_rewards = []
        self.range_events = []

    
    def evaluate_boundary(self, side):

        if side == "low":
            self.adr_wrapper.unwrapped.set_parameters(self.adr_wrapper.low)
        else:
            self.adr_wrapper.unwrapped.set_parameters(self.adr_wrapper.high)

        self.adr_wrapper.evaluation_mode = True
        mean_reward, _ = evaluate_policy(
            self.model,
            self.adr_wrapper,
            n_eval_episodes=5,
            deterministic=True
        )
        self.adr_wrapper.evaluation_mode = False
        
        return mean_reward

    def _on_step(self) -> bool:
        
        if self.n_calls % self.check_freq != 0:
            return True

        #self.nTimesteps += self.check_freq

        print("\n[ADR] Valutazione boundary")
        #print(f"Timesteps: {nTimesteps}")

        
        for i in range(0,len(self.adr_wrapper.nominal_masses)-1):
           
            low_reward = self.evaluate_boundary("low")
            print(f"[ADR] Param {i} | LOW reward = {low_reward:.2f}")

            if low_reward > self.up_thr:
                self.adr_wrapper.low[i] *= (1.0 - self.step)
                self.range_events.append((self.n_calls, "LOW-", i))
                print(f"[ADR] LOW[{i}] espanso")

            elif low_reward < self.low_thr:
                self.adr_wrapper.low[i] *= (1.0 + self.step)
                self.range_events.append((self.n_calls, "LOW+", i))
                print(f"[ADR] LOW[{i}] ristretto")


            high_reward = self.evaluate_boundary("high")
            print(f"[ADR] Param {i} | HIGH reward = {high_reward:.2f}")

            if high_reward > self.up_thr:
                self.adr_wrapper.high[i] *= (1.0 + self.step)
                self.range_events.append((self.n_calls, "HIGH+", i))
                print(f"[ADR] HIGH[{i}] espanso")

            elif high_reward < self.low_thr:
                self.adr_wrapper.high[i] *= (1.0 - self.step)
                self.range_events.append((self.n_calls, "HIGH-", i))
                print(f"[ADR] HIGH[{i}] ristretto")

        mean_reward = (low_reward + high_reward) / 2
        self.timesteps.append(self.n_calls)
        self.mean_rewards.append(mean_reward)

        print(f"[ADR] Mean reward boundary = {mean_reward:.2f}")

        
        max_range = np.max(self.adr_wrapper.high / self.adr_wrapper.nominal_masses[1:] - 1.0) #
        if max_range > 1.0:
            print("[ADR] Range > 1 raggiunto. Training interrotto.")
            return False

        return True

    
    
def train_agent(env, timesteps, adr_callback):
     model = PPO("MlpPolicy",env,learning_rate=0.0003, verbose = 0, gamma=0.99) #hyperameters
     model.learn(total_timesteps=timesteps, callback=adr_callback) #return a trained model
     return model

def test_agent(model, env):
    new_env = gym.make(env)
    print(f"--- Testing  ---")
    mean_reward, std_reward = evaluate_policy(model, new_env, n_eval_episodes=50, deterministic=True)
        
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print("-" * 30)
    return mean_reward
    

def plot_adr(callback):

    plt.figure(figsize=(10, 5))
    plt.plot(callback.timesteps, callback.mean_rewards, label="Mean Reward")

    for step, event, idx in callback.range_events:
        plt.axvline(step, linestyle="--", alpha=0.3)

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("ADR – Mean Reward & Range Updates")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    #parameters:
    initial_udr_range=0.0
    step_size = 0.05 #how much the range increases 
    performance_lower_bound=100.0 #mean reward could be into this interval, if overcame or is near to upper bound we update the range
    performance_upper_bound=600.0
    check_frequency = 30000 #defines how many timesteps should pass between performance evaluations

    base_env = gym.make("CustomHopper-source-v0")
    adr_env = HopperADR(env=base_env, initial_udr_range=initial_udr_range)


    adr_callback = HopperADRCallback(adr_wrapper=adr_env, check_freq=check_frequency, up_thr=performance_upper_bound, low_thr=performance_lower_bound, step=step_size)

    #Training and testing

    print("Training starts...")
    model = train_agent(env=base_env, timesteps=100000, adr_callback=adr_callback)
    model.save("Hopper ADR model")

    test_agent(model, "CustomHopper-source-v0")
    test_agent(model, "CustomHopper-target-v0")

if __name__ == "__main__":
    main()


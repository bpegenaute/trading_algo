import gym
import numpy as np
import pandas as pd
from gym import spaces

class TradingEnvironment(gym.Env):
    def __init__(self, data, initial_balance=10000, window_size=10):
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.current_step = 0

        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = spaces.Box(low=0, high=1, shape=(window_size, 6), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.inventory = []
        self.score = 0
        return self.get_observation()

    def get_observation(self):
        start = max(0, self.current_step - self.window_size)
        window = self.data.iloc[start:self.current_step]
        timestamp_column = 'Date' if 'Date' in window.columns else 'timestamp'
        obs = window.drop(columns=[timestamp_column]).to_numpy()
    
        # Pad the observation with zeros if it's smaller than the window_size
        pad_size = self.window_size - obs.shape[0]
        if pad_size > 0:
            obs = np.pad(obs, ((pad_size, 0), (0, 0)), mode='constant', constant_values=0)
    
        return obs

    def step(self, action):
        self.current_step += 1
        current_price = self.data.iloc[self.current_step]["Close"]

        reward = 0
        if action == 0:  # Buy
            self.inventory.append(current_price)
            self.balance -= current_price
            reward = -current_price
        elif action == 1:  # Sell
            if len(self.inventory) > 0:
                bought_price = self.inventory.pop(0)
                self.balance += current_price
                reward = current_price - bought_price
        else:
            pass  # Hold

        self.score += reward
        self.net_worth = self.balance + sum(self.inventory)
        done = self.net_worth <= 0 or self.current_step >= len(self.data) - 1
        info = {"net_worth": self.net_worth}

        return self.get_observation(), reward, done, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth}")

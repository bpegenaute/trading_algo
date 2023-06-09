import gym
import numpy as np
import pandas as pd
from gym import spaces
from sentiment import get_sentiment_score
from config import Config
from news_fetcher import fetch_news

config = Config()

class TradingEnvironment(gym.Env):
    def __init__(self, data, initial_balance=10000, window_size=10, validation_split=0.8):
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.current_step = 0
        self.validation_split = validation_split
        self.split_index = int(self.data.shape[0] * self.validation_split)
        self.train_data = self.data.iloc[:self.split_index]
        self.validation_data = self.data.iloc[self.split_index:]

        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = spaces.Box(low=0, high=1, shape=(window_size, 6), dtype=np.float32)

        self.inventory = []
        self.score = 0
        self.balance = self.initial_balance

    def _get_sentiment_scores(self):
        api_key = config.BING_API_KEY
        news_data = fetch_news(api_key, f'AAPL stock news')
        if news_data is not None:
            text_data = [article['name'] for article in news_data['value']]
            sentiment_scores = [get_sentiment_score(text) for text in text_data]
        else:
            sentiment_scores = []
        return sentiment_scores

    def _get_state(self, step):
        start = max(0, step - self.window_size)
        window = self.train_data.iloc[start:step]

        obs = window.to_numpy()

        # Pad the observation with zeros if it's smaller than the window_size
        pad_size = self.window_size - obs.shape[0]
        if pad_size > 0:
            obs = np.pad(obs, ((pad_size, 0), (0, 0)), mode='constant', constant_values=0)

        return obs

    def reset(self):
        self.current_step = self.window_size
        self.done = False
        self.profit = 0
        self.net_worths = [self.initial_balance]
        self.trades = []
        self.position = None
        self.position_price = None
        self.sentiment_scores = self._get_sentiment_scores()

        state = self._get_state(self.current_step)
        if len(self.sentiment_scores) > 0:
            sentiment_score = self.sentiment_scores[self.current_step - self.window_size]
        else:
            sentiment_score = 0  # If sentiment_scores is empty, assign a default value of 0

        return state, sentiment_score

    def reset_validation(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.inventory = []
        self.score = 0
        return self.get_observation(self.validation_data)

    def get_observation(self, data):
        start = max(0, self.current_step - self.window_size)
        window = data.iloc[start:self.current_step]
        
        obs = window.to_numpy()

        # Pad the observation with zeros if it's smaller than the window_size
        pad_size = self.window_size - obs.shape[0]
        if pad_size > 0:
            obs = np.pad(obs, ((pad_size, 0), (0, 0)), mode='constant', constant_values=0)

        return obs

    def step(self, action):
        return self._step(action, self.train_data)  # Call _step with the train_data

    def step_validation(self, action):
        return self._step(action, self.validation_data)

    def _step(self, action, data):
        self.current_step += 1
        current_price = data.iloc[self.current_step]["Close"]

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
        done = self.net_worth <= 0 or self.current_step >= len(data) - 1
        info = {"net_worth": self.net_worth}

        return self.get_observation(data), reward, done, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth}")

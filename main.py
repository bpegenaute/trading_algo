import os
import pandas as pd
import numpy as np
import torch
from config import *
from agent.environment import TradingEnvironment
from agent.model.lstm_model import LSTMModel
from agent.dqn_agent import DQNAgent
from agent.news_fetcher import fetch_news
from agent.sentiment import get_sentiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the trading environment
data = pd.read_csv('AAPL.csv')
env = TradingEnvironment(data)

# Parameters for the LSTM model
input_size = 61
hidden_size = 64
num_layers = 2
output_size = 3

# Instantiate the LSTM model and the target model
model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, device=device)
target_model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, device=device)

# Create the DQN agent
agent = DQNAgent(env, model, target_model, device)

# Training parameters
n_episodes = 1000
train_frequency = 10

for e in range(n_episodes):
    state = env.reset()
    done = False
    score = 0
    
    while not done:
        # Fetch the news
        news_data = fetch_news()
        # Perform sentiment analysis
        sentiment_scores = get_sentiment(news_data)

        action = agent.act(state, sentiment_scores)
        next_state, reward, done = env.step(action)
        agent.memorize(state, action, reward, next_state, done, sentiment_scores)
        
        state = next_state
        score += reward

    print(f"episode: {e}/{n_episodes}, score: {score}, e: {agent.epsilon}")

    agent.replay(32)
    # Retrain the model
    if e % train_frequency == 0:
        agent.retrain()

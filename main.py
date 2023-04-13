import os
import sys
import numpy as np
import yfinance as yf
from datetime import datetime
from environment.trading_env import TradingEnvironment
from dqn_agent import DQNAgent
from config import OPENAI_API_KEY, BING_API_KEY
from news_fetcher import fetch_news
from sentiment import analyze_sentiment
from openai_api import OpenAIAPI

symbol = "AAPL"
start_date = "2010-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

# Fetch historical stock data
data = yf.download(symbol, start=start_date, end=end_date)
data.reset_index(inplace=True)

# Fetch news articles
news = fetch_news(BING_API_KEY, symbol)

# Process news articles using OpenAI API
openai_api = OpenAIAPI(OPENAI_API_KEY)
news_summary = openai_api.generate_summary(news)

# Analyze sentiment
sentiment = analyze_sentiment(news_summary)

# Create the trading environment
env = TradingEnvironment(data, sentiment)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize the DQN agent
agent = DQNAgent(state_size, action_size)

# Train the agent
EPISODES = 1000
for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size, 6])
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size, 6])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"episode: {e}/{EPISODES}, score: {env.score}, e: {agent.epsilon:.2}")
            agent.update_target_model()
    if len(agent.memory) > 32:
        agent.replay(32)

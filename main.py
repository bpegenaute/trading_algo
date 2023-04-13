import os
import torch
import time
import yfinance as yf
import numpy as np
import datetime
from config import Config
from dqn_agent import DQNAgent
from environment import TradingEnvironment
from news_fetcher import fetch_news
from sentiment import get_sentiment_score
from openai_api import generate_text
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order

class IBApi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def error(self, reqId, errorCode, errorString):
        print(f"Error: {reqId} {errorCode} {errorString}")

    # Add more methods as needed for the IB API

def get_realtime_data(ticker, interval='1m', period='1d'):
    data = yf.download(ticker, interval=interval, period=period)
    return data

def preprocess_realtime_data(realtime_data, sentiment_score):
    # Normalize the data
    normalized_data = (realtime_data - realtime_data.min()) / (realtime_data.max() - realtime_data.min())

    # Reshape the data into the required format (batch_size, sequence_length, input_size)
    reshaped_data = np.reshape(normalized_data, (1, normalized_data.shape[0], normalized_data.shape[1]))

    # Combine the data with the sentiment score
    combined_data = np.concatenate((reshaped_data, np.full(reshaped_data.shape, sentiment_score)), axis=2)

    return torch.tensor(combined_data, dtype=torch.float32)

# Add a function to execute trade based on the predicted action
def execute_trade(action, ib_api):
    contract = Contract()
    contract.symbol = "AAPL"
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    contract.primaryExchange = "NASDAQ"

    order = Order()
    order.action = "BUY" if action == 1 else "SELL"
    order.totalQuantity = 100
    order.orderType = "MKT"

    orderId = 1  # You can use a unique order id for each order
    ib_api.placeOrder(orderId, contract, order)

if __name__ == "__main__":
    config = Config()
    env = TradingEnvironment(config=config)
    agent = DQNAgent(config=config, state_size=env.observation_space.shape[0], action_size=env.action_space.n)

    # Train the model using historical data
    for e in range(config.episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if len(agent.memory) > config.batch_size:
                agent.replay(config.batch_size)

        print(f"episode: {e}/{config.episodes}, score: {env.total_profit}, e: {agent.epsilon}")

    # Initialize the validation variables
    validation_net_worths = []
    validation_scores = []

    # Validate the model using validation data
    validation_state = env.reset_validation()
    done = False
    while not done:
        action = agent.act(validation_state)
        next_state, reward, done, info = env.step_validation(action)
        validation_state = next_state
        validation_scores.append(reward)
        validation_net_worths.append(info['net_worth'])

    # Calculate the average net worth and score during validation
    average_validation_net_worth = np.mean(validation_net_worths)
    average_validation_score = np.mean(validation_scores)

    print(f"Average validation net worth: {average_validation_net_worth}")
    print(f"Average validation score: {average_validation_score}")

    # Initialize the Interactive Brokers API
    ib_api = IBApi()
    ib_api.connect("127.0.0.1", config.IB_PORT, clientId=0)

    # Switch to the real-time trading mode
    while True:
        # Fetch the real-time stock data and news
        stock_data = get_realtime_data('AAPL')
        news_data = fetch_news('AAPL')

        # Calculate the sentiment score based on the news data
        sentiment_score = get_sentiment_score(news_data)

        # Preprocess the stock data and sentiment score for input to the model
        preprocessed_data = preprocess_realtime_data(stock_data, sentiment_score)

        # Predict the action using the trained model
        state = torch.tensor(preprocessed_data, dtype=torch.float32)
        action = agent.act(state)

        # Execute the trade using the Interactive Brokers API
        execute_trade(action, ib_api)

        # Sleep for a specified duration before fetching the next set of real-time data
        time.sleep(60)  # sleep for 1 minute
import os
import torch
import time
import yfinance as yf
import numpy as np
import datetime
from config import Config
from dqn_agent import DQNAgent
from environment.trading_env import TradingEnvironment
from news_fetcher import fetch_news
from sentiment import get_sentiment_score
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from dqn_model import DQNModel

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
    sentiment_data = np.full(reshaped_data.shape[:-1] + (1,), sentiment_score)
    combined_data = np.concatenate((reshaped_data, sentiment_data), axis=2)

    return torch.tensor(combined_data, dtype=torch.float32)

def get_historical_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

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
    historical_data = get_historical_data('AAPL', '2020-01-01', '2021-01-01')
    env = TradingEnvironment(data=historical_data, initial_balance=config.initial_balance, window_size=config.window_size)

    # Instantiate the DQN model and its target model
    model = DQNModel(state_size=config.state_size, action_size=config.action_size)
    target_model = DQNModel(state_size=config.state_size, action_size=config.action_size)

    # Instantiate the DQNAgent with the appropriate arguments
    agent = DQNAgent(
        state_size=config.state_size,
        action_size=config.action_size,
        memory_capacity=config.memory_capacity,
        gamma=config.gamma,
        epsilon=config.epsilon_start,
        epsilon_min=config.epsilon_min,
        epsilon_decay=config.epsilon_decay,
        learning_rate=config.learning_rate,
        model=model,
        target_model=target_model,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Train the model using historical data
    for e in range(config.episodes):
        state = env.reset()
        sentiment_scores = np.array(state[:-1])
        sentiment_score = sentiment_scores.mean()
        total_profit = 0
        done = False

        print(f"Starting training episode {e + 1}")

        while not done:
            action = agent.act(state, config.action_size)

            # Fetch the news and calculate sentiment score for the current step
            api_key = config.BING_API_KEY
            news_data = fetch_news(api_key, f'AAPL stock news')
            print(f"Fetched news: {news_data}")

            text_data = [article['name'] for article in news_data['value']]
            sentiment_scores = [get_sentiment_score(text) for text in text_data]
            sentiment_score = np.mean(sentiment_scores)

            print(f"Sentiment scores: {sentiment_scores}")
            print(f"Average sentiment score: {sentiment_score}")

            next_state, reward, done, _ = env.step(action)
            next_state_with_sentiment = np.hstack((next_state, np.full((next_state.shape[0], 1), sentiment_score)))
            next_state = torch.tensor(next_state_with_sentiment, dtype=torch.float32)

            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if len(agent.memory) > config.batch_size:
                agent.replay(config.batch_size)

            print(f"Training trade: Action: {action}, Reward: {reward}")

        print(f"End of training episode {e + 1}: Total profit: {env.total_profit}, Epsilon: {agent.epsilon}")

    # Initialize the validation variables
    validation_net_worths = []
    validation_scores = []

    # Validate the model using validation data
    validation_state = env.reset_validation()
    done = False
    print("Starting validation")

    while not done:
        action = agent.act(validation_state)

        # Fetch the news and calculate sentiment score for the current step
        api_key = config.BING_API_KEY
        news_data = fetch_news(api_key, f'AAPL stock news')
        print(f"Fetched news: {news_data}")

        text_data = [article['name'] for article in news_data['value']]
        sentiment_scores = [get_sentiment_score(text) for text in text_data]
        sentiment_score = np.mean(sentiment_scores)

        print(f"Sentiment scores: {sentiment_scores}")
        print(f"Average sentiment score: {sentiment_score}")

        next_state, reward, done, info = env.step_validation(action)
        next_state = torch.tensor(np.concatenate((next_state, [sentiment_score])), dtype=torch.float32)

        validation_state = next_state
        validation_scores.append(reward)
        validation_net_worths.append(info['net_worth'])

        print(f"Validation trade: Action: {action}, Reward: {reward}")

    # Calculate the average net worth and score during validation
    average_validation_net_worth = np.mean(validation_net_worths)
    average_validation_score = np.mean(validation_scores)

    print(f"Average validation net worth: {average_validation_net_worth}")
    print(f"Average validation score: {average_validation_score}")

    # Initialize the Interactive Brokers API
    ib_api = IBApi()
        ib_api.connect("127.0.0.1", config.IB_PORT, clientId=0)

    # Switch to the real-time trading mode
    print("Starting real-time trading")
    while True:
        # Fetch the real-time stock data and news
        stock_data = get_realtime_data('AAPL')
        print(f"Real-time stock data: {stock_data}")

        api_key = config.BING_API_KEY
        news_data = fetch_news(api_key, f'AAPL stock news')
        print(f"Fetched news: {news_data}")

        # Extract relevant text (e.g., news headlines) from fetched news articles
        text_data = [article['name'] for article in news_data['value']]

        # Calculate the sentiment scores and their mean for the fetched news articles
        sentiment_scores = [get_sentiment_score(text) for text in text_data]
        sentiment_score = np.mean(sentiment_scores)

        print(f"Sentiment scores: {sentiment_scores}")
        print(f"Average sentiment score: {sentiment_score}")

        # Preprocess the stock data and sentiment score for input to the model
        preprocessed_data = preprocess_realtime_data(stock_data, sentiment_score)

        # Predict the action using the trained model
        state = torch.tensor(preprocessed_data, dtype=torch.float32)
        action = agent.act(state, config.action_size)

        print(f"Predicted action: {action}")

        # Execute the trade using the Interactive Brokers API
        execute_trade(action, ib_api)

        # Sleep for a specified duration before fetching the next set of real-time data
        time.sleep(60)  # sleep for 1 minute
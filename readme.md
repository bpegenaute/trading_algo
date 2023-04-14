# Trading Algorithm

This project aims to develop an automated trading bot that uses deep reinforcement learning to execute trades based on real-time stock data, news sentiment, and technical indicators. The bot is designed to work with the Interactive Brokers API, Yahoo Finance API, Bing News API, and OpenAI API.

## Features

- Implements a DQN agent with LSTM neural network architecture for reinforcement learning
- Trains the model using historical stock data
- Fetches real-time stock data using Yahoo Finance API
- Retrieves and processes recent news articles related to the stock using Bing News API
- Calculates news sentiment scores using OpenAI API
- Executes trades automatically through the Interactive Brokers API
- Continuously improves its strategy by retraining based on trade results

## Installation and Setup

1. Clone the repository:

git clone https://github.com/bpegenaute/trading_algo.git

2. Create a virtual environment and activate it:

python3.9 -m venv venv
source venv/bin/activate

3. Install the required dependencies:

pip install -r requirements.txt

4. Create a `.env` file in the root directory of the project and add the following keys with appropriate values:

IB_ACCOUNT=<Your Interactive Brokers Account ID>
IB_PORT=7496
BING_API_KEY=<Your Bing News API Key>
OPENAI_API_KEY=<Your OpenAI API Key>
ALPHA_VANTAGE_API_KEY=<Your Alpha Vantage API Key>

## Running the Trading Bot

1. Run the `main.py` script to start the trading bot:

python main.py

2. The bot will first train the model using historical stock data and then switch to real-time trading mode, executing trades based on real-time data, news sentiment, and technical indicators.

3. The bot continuously improves its strategy by retraining based on trade results.

## Dependencies // See requirements.txt for latest versions

- Python 3.9
- PyTorch 1.9.1
- torchvision 0.10.1
- numpy 1.21.2
- pandas 1.3.3
- yfinance 0.1.63
- ibapi 9.80.20210824
- requests 2.26.0
- openai 0.27.0
- gym ##check version 
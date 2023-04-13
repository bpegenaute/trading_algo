# AI-Powered Stock Trading Bot

This is an AI-powered stock trading bot that uses reinforcement learning and the Interactive Brokers API to execute trades. The bot aims to learn and improve its trading strategy over time.

## Requirements

- Python 3.9 or higher
- Interactive Brokers account with API access
- Interactive Brokers Gateway installed and running

## Dependencies

- ib_insync
- numpy
- pandas
- torch

To install the dependencies, run the following command:

pip install ib_insync numpy pandas torch


## Setup

1. Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate # On Windows, use venv\Scripts\activate


2. Install the required libraries:

pip install ib_insync numpy pandas torch


3. Set up the `config.py` file with your Interactive Brokers account information:


# Change this line for a paper trading account
IB_ACCOUNT = "DUxxxxx"

# Change this line for a real account
# IB_ACCOUNT = "Uxxxxx"

# Port number for IB Gateway
IB_PORT = 7496

4. Launch the IB Gateway application, log in with your API credentials, and make sure the port number matches the one in the config.py file.
Running the Trading Bot
To run the trading bot, execute the following command:


python main.py

This will start the training process and connect to the Interactive Brokers API to place trades.

Disclaimer
Trading stocks and other assets carry inherent risks. Use this trading bot at your own risk. The performance of the trading strategy is not guaranteed, and you should thoroughly evaluate its effectiveness and understand the associated risks before using it with real funds.# trading_algo

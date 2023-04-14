import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self):
        self.episodes = 1000
        self.window_size = 10
        self.batch_size = 32
        self.memory_capacity = 1000000
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.lr = 0.001
        self.target_network_update_interval = 1000
        self.IB_PORT = int(os.getenv("IB_PORT"))
        self.initial_balance = 10000
        self.IB_ACCOUNT = os.getenv("IB_ACCOUNT")
        self.BING_API_KEY = os.getenv("BING_API_KEY")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

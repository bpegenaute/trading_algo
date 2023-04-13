import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    IB_ACCOUNT = os.getenv("IB_ACCOUNT")
    IB_PORT = int(os.getenv("IB_PORT"))
    BING_API_KEY = os.getenv("BING_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
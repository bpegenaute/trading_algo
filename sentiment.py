from textblob import TextBlob
from openai_api import OpenAIAPI
from config import Config
import openai

config = Config()
openai_api = OpenAIAPI(config.OPENAI_API_KEY)

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def get_sentiment_score(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert financial and news analyst with access to all the information avaialble up to 2021. Give me the sentiment score of the following text, be very exact, answer must be a number, nothing else: '{text}"},
            ],
    )
    score = response.choices[0]
    return score

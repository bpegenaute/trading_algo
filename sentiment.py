import os
from textblob import TextBlob
from openai_api import OpenAIAPI

openai_api = OpenAIAPI(api_key=os.environ['OPENAI_API_KEY'])

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def get_sentiment_score(text):
    rresponse = openai_api.get_sentiment_score(text)
    score = rresponse
    return score


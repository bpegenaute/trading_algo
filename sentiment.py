from textblob import TextBlob

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def get_sentiment_score(text):
    prompt = f"Sentiment score of the following text: '{text}'.\n\nScore:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=3,
        n=1,
        stop=None,
        temperature=0.7,
    )
    score = float(response.choices[0].text.strip())
    return scores
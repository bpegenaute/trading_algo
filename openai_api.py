import openai
import re
from urllib.parse import quote_plus

class OpenAIAPI:
    def __init__(self, api_key):
        openai.api_key = api_key

    def get_sentiment_score(self, text):
        prompt = f"Please analyze the sentiment of the following text and return only the sentiment score as a number between -1 (very negative) and 1 (very positive), neutral being 0, think this step by step. Your exact and only allowed answer should be in the form of: 'Sentiment: x.xxxx'. Where 'x.xxxx' is the number of the sentiment score. Absolutely nothing else than this format in your answer is allowed:\n\n{quote_plus(text)}"
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": "You're a the most performing tool ever seen doing sentiment analysis that considers billion+ parameters in a very methodical way every time and give accurate numerical results. Think step by step, but only answer a number."}, {"role": "user", "content": prompt}],
                    max_tokens=1000,
                    n=1,
                    stop=None,
                    temperature=0,
                )
                content = response['choices'][0]['message']['content'].strip()
                match = re.search(r"Sentiment[:\s]*([\-0-9\.]+)", content)
                if match:
                    score = float(match.group(1))
                    return score
                else:
                    print(f"Failed to extract sentiment score from content: {content}")
                    attempt += 1
            except openai.error.Timeout as e:
                print(f"Request timed out (attempt {attempt + 1}): {e}")
                attempt += 1

        print(f"Failed to get sentiment score after {max_attempts} attempts.")
        return 0.0

    def generate_summary(self, text, model='text-davinci-002', max_tokens=1000):
        prompt = f"Please summarize the following text in {max_tokens} words or less: '{text}'"
        response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=max_tokens, n=1, stop=None, temperature=0)
        return response.choices[0].text.strip()

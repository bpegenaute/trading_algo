import openai

class OpenAIAPI:
    def __init__(self, api_key):
        openai.api_key = api_key

    def generate_summary(text):
        openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert financial and news analyst with access to all the information avaialble up to 2021. Please summarize the following text in {max_tokens} words or less and : '{text}'"},
            ],
        )
        
        return response.choices[0]
        
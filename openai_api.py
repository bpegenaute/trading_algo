import openai

class OpenAIAPI:
    def __init__(self, api_key):
        openai.api_key = api_key

    def generate_summary(self, text, model='text-davinci-002', max_tokens=50):
        prompt = f"Please summarize the following text in {max_tokens} words or less: '{text}'"
        response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=max_tokens, n=1, stop=None, temperature=0.5)
        return response.choices[0].text.strip()

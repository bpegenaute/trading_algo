import requests

def fetch_news(api_key, query, count=10):
    url = f'https://api.bing.microsoft.com/v7.0/news/search?q={query}&count={count}&freshness=Day'
    headers = {'Ocp-Apim-Subscription-Key': api_key}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f'Error fetching news: {response.status_code}')
        return None
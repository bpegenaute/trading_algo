import requests

def fetch_news(api_key, query, count=1000):
    url = f'https://api.bing.microsoft.com/v7.0/news/search?q={query}&count={count}&freshness=Day'
    headers = {'Ocp-Apim-Subscription-Key': api_key}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f'Error fetching news: {response.status_code}')
        return None

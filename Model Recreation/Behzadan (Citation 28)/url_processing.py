import requests

def get_final_destination_url(shortened_url):
    try:
        response = requests.head(shortened_url, allow_redirects=True)
        return response.url
    except requests.exceptions.RequestException:
        return None

def process_url(url):
    final_url = get_final_destination_url(url)
    return final_url
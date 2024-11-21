# app/utils.py

from urllib.parse import urlparse

def is_valid_url(url):
    """
    Checks if a URL is valid.
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

# tests/test_utils.py

import pytest
from app.utils import is_valid_url

def test_is_valid_url():
    """
    Test the URL validation function with various inputs.
    """
    # Valid URLs
    valid_urls = [
        "https://www.example.com",
        "http://sub.domain.example.co.uk/path?query=param#fragment",
        "https://localhost:8000",
        "https://127.0.0.1",
        "https://www.example.com:8080",
    ]
    
    for url in valid_urls:
        assert is_valid_url(url) == True, f"URL should be valid: {url}"
    
    # Invalid URLs
    invalid_urls = [
        "ftp://invalid-protocol.com",
        "www.example.com",  # Missing scheme
        "http//missing-colon.com",
        "://missing-scheme.com",
        "justastring",
        "",
        "http:/single-slash.com",
        "http://",
    ]
    
    for url in invalid_urls:
        assert is_valid_url(url) == False, f"URL should be invalid: {url}"

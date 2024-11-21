# tests/test_processors.py

import pytest
from app.processors import URLProcessor, TextProcessor

def test_is_valid_url():
    processor = URLProcessor([])
    assert processor.is_valid_url("https://www.example.com") == True
    assert processor.is_valid_url("invalid-url") == False

def test_url_validation():
    urls = ["https://www.validurl.com", "ftp://invalidprotocol.com", "not-a-url"]
    processor = URLProcessor(urls)
    valid, invalid = processor.validate_urls()
    assert valid == ["https://www.validurl.com"]
    assert invalid == ["ftp://invalidprotocol.com", "not-a-url"]

def test_text_splitter():
    data = [{"text": "This is a test document. It has multiple sentences."}]
    text_processor = TextProcessor(data)
    docs = text_processor.split_text()
    assert len(docs) == 1
    assert docs[0].page_content == "This is a test document. It has multiple sentences."

# app/processors.py

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.utils import is_valid_url
import logging

class URLProcessor:
    """
    Handles URL validation, loading, and processing.
    """

    def __init__(self, urls):
        self.urls = urls
        self.valid_urls = []
        self.invalid_urls = []

    def validate_urls(self):
        """
        Validates the list of URLs.
        """
        for url in self.urls:
            if is_valid_url(url):
                self.valid_urls.append(url)
            else:
                self.invalid_urls.append(url)
        return self.valid_urls, self.invalid_urls

    def load_data(self):
        """
        Loads data from valid URLs.
        """
        loader = UnstructuredURLLoader(urls=self.valid_urls)
        data = loader.load()
        logging.info(f"Loaded data from URLs: {self.valid_urls}")
        return data

class TextProcessor:
    """
    Handles text splitting and summarization.
    """

    def __init__(self, data):
        self.data = data
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        self.docs = []

    def split_text(self):
        """
        Splits the loaded data into manageable chunks.
        """
        self.docs = self.text_splitter.split_documents(self.data)
        logging.info("Text splitting completed.")
        return self.docs

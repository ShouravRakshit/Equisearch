# app/embeddings.py

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import logging

class EmbeddingManager:
    """
    Handles creation and storage of embeddings.
    """

    def __init__(self, docs):
        self.docs = docs
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None

    def create_embeddings(self):
        """
        Creates embeddings from documents.
        """
        self.vectorstore = FAISS.from_documents(self.docs, self.embeddings)
        logging.info("Embeddings created.")
        return self.vectorstore

    def save_embeddings(self, file_path):
        """
        Saves the FAISS vector store locally.
        """
        self.vectorstore.save_local(file_path)
        logging.info(f"Embeddings saved to {file_path}.")

    @staticmethod
    def load_embeddings(file_path):
        """
        Loads the FAISS vector store from local storage.
        """
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
        logging.info(f"Embeddings loaded from {file_path}.")
        return vectorstore

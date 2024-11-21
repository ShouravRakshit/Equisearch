# app/main.py

import sys
import os
import streamlit as st
import time
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
import pandas as pd

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.processors import URLProcessor, TextProcessor
from app.embeddings import EmbeddingManager
from app.qa import QAChain
from app.logging_config import setup_logging
import logging

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()

class EQUISEARCHApp:
    """
    Main application class for EQUISEARCH: News Research Tool.
    """

    def __init__(self):
        self.file_path = "faiss_store_openai"
        self.llm = OpenAI(temperature=0.5, max_tokens=500)
        self.setup_ui()

    def setup_ui(self):
        """
        Sets up the Streamlit user interface.
        """
        st.set_page_config(page_title="EQUISEARCH: News Research Tool 📈", layout="wide")
        st.title("EQUISEARCH: News Research Tool 📈")
        st.sidebar.title("News Article URLs")

        # Bulk URL input via textarea
        bulk_urls = st.sidebar.text_area("Enter URLs (one per line):")
        urls = [url.strip() for url in bulk_urls.split('\n') if url.strip()]

        process_url_clicked = st.sidebar.button("Process URLs")

        self.main_placeholder = st.empty()

        if process_url_clicked:
            self.process_urls(urls)

        # Query section
        query = st.text_input("Question: ")
        if query:
            self.handle_query(query)

    @st.cache_data(show_spinner=False)
    def load_and_process_urls(self, urls):
        """
        Caches the loading and processing of URLs.
        """
        processor = URLProcessor(urls)
        valid_urls, invalid_urls = processor.validate_urls()
        if invalid_urls:
            st.error(f"Invalid URLs detected: {invalid_urls}")
            logging.warning(f"Invalid URLs: {invalid_urls}")
        if not valid_urls:
            st.error("No valid URLs to process.")
            logging.error("No valid URLs provided.")
            return None

        data = processor.load_data()
        text_processor = TextProcessor(data)
        docs = text_processor.split_text()
        return docs

    def process_urls(self, urls):
        """
        Processes the provided URLs: validation, loading, splitting, embedding, and saving.
        """
        processor = URLProcessor(urls)
        valid_urls, invalid_urls = processor.validate_urls()

        if invalid_urls:
            st.error(f"Invalid URLs detected: {invalid_urls}")
            logging.warning(f"Invalid URLs: {invalid_urls}")
        
        if not valid_urls:
            st.error("No valid URLs to process.")
            logging.error("No valid URLs provided.")
            return

        with st.spinner("Loading data..."):
            data = processor.load_data()
        
        with st.spinner("Splitting text..."):
            text_processor = TextProcessor(data)
            docs = text_processor.split_text()
        
        with st.spinner("Building embeddings..."):
            embedding_manager = EmbeddingManager(docs)
            vectorstore = embedding_manager.create_embeddings()
            embedding_manager.save_embeddings(self.file_path)
        
        with st.spinner("Generating summaries..."):
            summarizer = load_summarize_chain(self.llm, chain_type="map_reduce")
            summaries = summarizer.run(docs)
            st.subheader("Article Summaries")
            st.write(summaries)
            logging.info("Summarization completed.")
        
        st.success("Processing completed successfully!")
        logging.info("URL processing completed.")

    def handle_query(self, query):
        """
        Handles user queries by retrieving answers and sources.
        """
        if os.path.exists(self.file_path):
            with st.spinner("Loading embeddings..."):
                vectorstore = EmbeddingManager.load_embeddings(self.file_path)
            
            with st.spinner("Generating answer..."):
                qa_chain = QAChain(self.llm, vectorstore)
                answer, sources = qa_chain.get_answer(query)
                logging.info(f"Query received: {query}")
            
            st.header("Answer")
            st.write(answer)
            
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
        else:
            st.error("Embeddings not found. Please process URLs first.")
            logging.error("Embeddings file not found when attempting to query.")

if __name__ == "__main__":
    app = EQUISEARCHApp()

# app.py

import os
import asyncio
import streamlit as st
import pickle
import time
from dotenv import load_dotenv

# Import necessary libraries for NLP and Data Visualization
import plotly.express as px
import pandas as pd
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import OpenAI and LangChain modules
from langchain import OpenAI, PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Import NLP libraries
import spacy
from textblob import TextBlob

# Load environment variables
load_dotenv()

# Initialize NLP models
nlp = spacy.load("en_core_web_sm")  # Make sure to download the model: python -m spacy download en_core_web_sm

# Initialize OpenAI API
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(openai_api_key=openai_api_key, temperature=0.5, max_tokens=500)

# Set up Streamlit page configuration
st.set_page_config(page_title="EQUISEARCH: News Research Tool", layout="wide")

# Title and Sidebar
st.title("EQUISEARCH: News Research Tool 📈")
st.sidebar.title("Options")

# Sidebar - Input Method Selection
input_method = st.sidebar.selectbox("Select Input Method", ["URLs", "Upload Files", "Search News API"])

# Initialize variables
docs = []

# Function to load and process data from URLs
@st.cache_data
def load_data_from_urls(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    return data

# Function to process text data
def process_data(data):
    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(data)
    return docs

# Function to generate embeddings and create vectorstore
def create_vectorstore(docs):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Function to perform sentiment analysis
def perform_sentiment_analysis(docs):
    sentiments = []
    for doc in docs:
        blob = TextBlob(doc.page_content)
        sentiments.append(blob.sentiment.polarity)
    return sentiments

# Function to perform Named Entity Recognition
def perform_ner(docs):
    entities = []
    for doc in docs:
        spacy_doc = nlp(doc.page_content)
        entities.extend([(ent.text, ent.label_) for ent in spacy_doc.ents])
    return entities

# Function to perform topic modeling
def perform_topic_modeling(docs):
    texts = [doc.page_content for doc in docs]
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    # Use TruncatedSVD for dimensionality reduction
    from sklearn.decomposition import TruncatedSVD
    svd_model = TruncatedSVD(n_components=5, random_state=42)
    svd_model.fit(X)
    terms = vectorizer.get_feature_names_out()
    topics = []
    for i, comp in enumerate(svd_model.components_):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:10]
        topics.append([t[0] for t in sorted_terms])
    return topics

# Function to visualize embeddings
def visualize_embeddings(vectorstore, docs):
    vectors = vectorstore.get_vectors()
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(vectors)
    df = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
    df['Document'] = [doc.metadata.get('source', 'Unknown') for doc in docs]
    fig = px.scatter(df, x='x', y='y', hover_data=['Document'])
    st.plotly_chart(fig)

# Input Methods
if input_method == "URLs":
    urls = st.sidebar.text_area("Enter URLs (one per line)").splitlines()
    if st.sidebar.button("Process URLs"):
        with st.spinner("Loading and processing data..."):
            data = load_data_from_urls(urls)
            docs = process_data(data)
            vectorstore = create_vectorstore(docs)
            st.success("Data processed successfully!")
elif input_method == "Upload Files":
    uploaded_files = st.sidebar.file_uploader("Upload Documents", accept_multiple_files=True)
    if st.sidebar.button("Process Files"):
        with st.spinner("Loading and processing data..."):
            data = []
            for uploaded_file in uploaded_files:
                content = uploaded_file.read().decode('utf-8', errors='ignore')
                metadata = {'source': uploaded_file.name}
                data.append({'page_content': content, 'metadata': metadata})
            docs = process_data(data)
            vectorstore = create_vectorstore(docs)
            st.success("Data processed successfully!")
elif input_method == "Search News API":
    api_key = st.sidebar.text_input("Enter News API Key")
    query = st.sidebar.text_input("Enter Search Query")
    if st.sidebar.button("Search and Process"):
        with st.spinner("Fetching and processing data..."):
            import requests
            url = f'https://newsapi.org/v2/everything?q={query}&apiKey={api_key}'
            response = requests.get(url)
            articles = response.json().get('articles', [])
            data = []
            for article in articles:
                content = article.get('content', '')
                metadata = {'source': article.get('url', 'Unknown')}
                data.append({'page_content': content, 'metadata': metadata})
            docs = process_data(data)
            vectorstore = create_vectorstore(docs)
            st.success("Data processed successfully!")

# Display Data Visualization Options
if docs:
    st.subheader("Data Visualization")
    if st.checkbox("Show Embedding Visualization"):
        visualize_embeddings(vectorstore, docs)

    if st.checkbox("Show Sentiment Analysis"):
        sentiments = perform_sentiment_analysis(docs)
        st.bar_chart(sentiments)

    if st.checkbox("Show Named Entities"):
        entities = perform_ner(docs)
        entity_counts = Counter([ent[0] for ent in entities])
        st.write(entity_counts.most_common(10))

    if st.checkbox("Show Topic Modeling"):
        topics = perform_topic_modeling(docs)
        for i, topic in enumerate(topics):
            st.write(f"Topic {i+1}: {', '.join(topic)}")

    # Save the vectorstore to disk
    vectorstore.save_local("vectorstore")

# Question-Answering Section
if 'vectorstore' in locals():
    st.subheader("Ask a Question")
    user_question = st.text_input("Enter your question")
    if user_question:
        retriever = vectorstore.as_retriever()
        # Custom prompt template
        prompt_template = """
        You are an expert news analyst. Use the following context to answer the question.

        Context:
        {context}

        Question:
        {question}

        Answer in a clear and concise manner, citing sources when appropriate.
        """
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs
        )
        with st.spinner("Generating answer..."):
            result = chain({"question": user_question}, return_only_outputs=True)
            st.write("### Answer")
            st.write(result["answer"])

            # Display sources
            sources = result.get("sources", "")
            if sources:
                st.write("### Sources")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)

    # Feedback mechanism
    st.subheader("Feedback")
    rating = st.slider("Rate the answer", 1, 5)
    if st.button("Submit Rating"):
        st.success("Thank you for your feedback!")

# Additional Features
if docs:
    st.subheader("Additional Features")

    if st.checkbox("Show Article Summaries"):
        from transformers import pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summaries = [summarizer(doc.page_content, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] for doc in docs]
        for i, summary in enumerate(summaries):
            st.write(f"**Summary of Document {i+1}:**")
            st.write(summary)

    if st.checkbox("Compare Articles"):
        st.write("Calculating similarity matrix...")
        texts = [doc.page_content for doc in docs]
        tfidf = TfidfVectorizer().fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf)
        df_sim = pd.DataFrame(similarity_matrix)
        st.write(df_sim)

# Cache function for performance optimization
@st.cache_data
def expensive_computation(a, b):
    time.sleep(2)  # Simulate a long computation
    return a + b

# Example of using the cache
# result = expensive_computation(2, 21)
# st.write(f"The result is {result}")


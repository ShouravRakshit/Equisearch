# tests/test_embeddings.py

import pytest
from unittest.mock import patch, MagicMock
from app.embeddings import EmbeddingManager

@pytest.fixture
def sample_docs():
    return [
        {"page_content": "Sample text for embedding one."},
        {"page_content": "Another sample text for embedding two."}
    ]

@patch('app.embeddings.OpenAIEmbeddings')
@patch('app.embeddings.FAISS')
def test_create_embeddings(mock_faiss, mock_openai_embeddings, sample_docs):
    """
    Test the creation of embeddings.
    """
    # Mock the embeddings instance
    mock_embeddings_instance = MagicMock()
    mock_openai_embeddings.return_value = mock_embeddings_instance
    
    # Mock the FAISS.from_documents method
    mock_vectorstore = MagicMock()
    mock_faiss.from_documents.return_value = mock_vectorstore
    
    embedding_manager = EmbeddingManager(sample_docs)
    vectorstore = embedding_manager.create_embeddings()
    
    # Assertions
    mock_openai_embeddings.assert_called_once()
    mock_faiss.from_documents.assert_called_once_with(sample_docs, mock_embeddings_instance)
    assert vectorstore == mock_vectorstore

@patch('app.embeddings.FAISS')
def test_save_embeddings(mock_faiss, sample_docs):
    """
    Test saving embeddings to a local file.
    """
    # Mock the embeddings instance and vectorstore
    mock_vectorstore = MagicMock()
    mock_faiss.from_documents.return_value = mock_vectorstore
    
    embedding_manager = EmbeddingManager(sample_docs)
    embedding_manager.embeddings = MagicMock()  # Mock embeddings to avoid actual calls
    embedding_manager.vectorstore = mock_vectorstore
    
    file_path = "test_faiss_store"
    embedding_manager.save_embeddings(file_path)
    
    # Assertions
    mock_vectorstore.save_local.assert_called_once_with(file_path)

@patch('app.embeddings.FAISS')
@patch('app.embeddings.OpenAIEmbeddings')
def test_load_embeddings(mock_openai_embeddings, mock_faiss, sample_docs):
    """
    Test loading embeddings from a local file.
    """
    # Mock the embeddings instance and FAISS.load_local method
    mock_embeddings_instance = MagicMock()
    mock_openai_embeddings.return_value = mock_embeddings_instance
    
    mock_loaded_vectorstore = MagicMock()
    mock_faiss.load_local.return_value = mock_loaded_vectorstore
    
    file_path = "test_faiss_store"
    vectorstore = EmbeddingManager.load_embeddings(file_path)
    
    # Assertions
    mock_openai_embeddings.assert_called_once()
    mock_faiss.load_local.assert_called_once_with(file_path, mock_embeddings_instance, allow_dangerous_deserialization=True)
    assert vectorstore == mock_loaded_vectorstore

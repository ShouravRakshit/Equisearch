# tests/test_qa.py

import pytest
from unittest.mock import patch, MagicMock
from app.qa import QAChain

@patch('app.qa.RetrievalQAWithSourcesChain')
def test_get_answer(mock_retrieval_qa_chain):
    """
    Test retrieving an answer and sources for a given question.
    """
    # Setup mock chain
    mock_chain_instance = MagicMock()
    mock_retrieval_qa_chain.from_llm.return_value = mock_chain_instance
    mock_chain_instance.__call__.return_value = {
        "answer": "This is a mocked answer.",
        "sources": "Source1\nSource2"
    }
    
    # Initialize QAChain
    llm = MagicMock()
    vectorstore = MagicMock()
    qa_chain = QAChain(llm, vectorstore)
    
    # Call get_answer
    question = "What is the capital of France?"
    answer, sources = qa_chain.get_answer(question)
    
    # Assertions
    mock_retrieval_qa_chain.from_llm.assert_called_once_with(llm=llm, retriever=vectorstore.as_retriever())
    mock_chain_instance.__call__.assert_called_once_with({"question": question}, return_only_outputs=True)
    assert answer == "This is a mocked answer."
    assert sources == "Source1\nSource2"
